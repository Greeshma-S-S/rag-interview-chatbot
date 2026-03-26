# AWS Deployment Guide — AI Engineer RAG Chatbot

This guide covers three deployment approaches, from simple (EC2) to scalable (ECS Fargate).

---

## Prerequisites

- AWS CLI configured (`aws configure`)
- Docker installed locally
- An AWS account with appropriate IAM permissions
- Your `.env` file ready with `OPENAI_API_KEY`

---

## Option 1: EC2 (Quickest — Dev/Staging)

### Step 1: Launch an EC2 instance

```bash
# Launch Ubuntu 22.04 t3.medium (min 4 GB RAM for FAISS + models)
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.medium \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxx \
  --subnet-id subnet-xxxxxxxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=rag-chatbot}]' \
  --count 1
```

### Step 2: Configure security group

```bash
# Allow inbound on ports 22 (SSH), 8000 (API), 8501 (Streamlit)
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxx \
  --protocol tcp --port 22 --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxx \
  --protocol tcp --port 8000 --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxx \
  --protocol tcp --port 8501 --cidr 0.0.0.0/0
```

### Step 3: Install Docker on EC2

```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

# Install Docker
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker ubuntu
newgrp docker
```

### Step 4: Deploy the application

```bash
# Clone or upload your code
git clone https://github.com/your-repo/rag-chatbot.git
cd rag-chatbot

# Create .env file
cp .env.example .env
nano .env  # add your OPENAI_API_KEY

# Download papers and build vector store
docker compose -f docker/docker-compose.yml run --rm api \
  python scripts/download_pdfs.py

docker compose -f docker/docker-compose.yml run --rm api \
  python scripts/ingest_data.py

# Start all services
docker compose -f docker/docker-compose.yml up -d

# Check logs
docker compose -f docker/docker-compose.yml logs -f api
```

### Step 5: Access the app

- API: `http://<EC2_PUBLIC_IP>:8000`
- Swagger UI: `http://<EC2_PUBLIC_IP>:8000/docs`
- Streamlit: `http://<EC2_PUBLIC_IP>:8501`

---

## Option 2: ECS Fargate (Production — Recommended)

### Architecture

```
Internet → ALB (HTTPS) → ECS Fargate (API + Frontend)
                              ↓
                         S3 (PDFs) + EFS (Vector Store)
                              ↓
                    AWS Secrets Manager (OPENAI_API_KEY)
```

### Step 1: Create ECR repositories

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create repositories
aws ecr create-repository --repository-name rag-chatbot-api --region $AWS_REGION
aws ecr create-repository --repository-name rag-chatbot-frontend --region $AWS_REGION

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS \
  --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
```

### Step 2: Build and push Docker images

```bash
# Build API image
docker build -f docker/Dockerfile -t rag-chatbot-api:latest .
docker tag rag-chatbot-api:latest \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/rag-chatbot-api:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/rag-chatbot-api:latest

# Build Frontend image
docker build -f docker/Dockerfile.frontend -t rag-chatbot-frontend:latest .
docker tag rag-chatbot-frontend:latest \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/rag-chatbot-frontend:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/rag-chatbot-frontend:latest
```

### Step 3: Store secrets in AWS Secrets Manager

```bash
# Store OpenAI API key
aws secretsmanager create-secret \
  --name "/rag-chatbot/openai-api-key" \
  --description "OpenAI API Key for RAG Chatbot" \
  --secret-string '{"OPENAI_API_KEY":"sk-your-key-here"}'
```

### Step 4: Create EFS for persistent vector store

```bash
# Create EFS filesystem
EFS_ID=$(aws efs create-file-system \
  --creation-token rag-vectorstore \
  --performance-mode generalPurpose \
  --query FileSystemId --output text)

echo "EFS ID: $EFS_ID"

# Create mount target (in your VPC subnet)
aws efs create-mount-target \
  --file-system-id $EFS_ID \
  --subnet-id subnet-xxxxxxxx \
  --security-groups sg-xxxxxxxx
```

### Step 5: Create ECS Cluster and Task Definition

```bash
# Create cluster
aws ecs create-cluster --cluster-name rag-chatbot-cluster

# Register task definition (save as ecs-task-definition.json first)
aws ecs register-task-definition \
  --cli-input-json file://aws/ecs-task-definition.json
```

**`aws/ecs-task-definition.json`** (save this file):

```json
{
  "family": "rag-chatbot",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "rag-api",
      "image": "ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/rag-chatbot-api:latest",
      "portMappings": [{"containerPort": 8000, "protocol": "tcp"}],
      "secrets": [
        {"name": "OPENAI_API_KEY", "valueFrom": "/rag-chatbot/openai-api-key:OPENAI_API_KEY::"}
      ],
      "mountPoints": [
        {"sourceVolume": "vectorstore", "containerPath": "/app/data/vectorstore"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/rag-chatbot",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "api"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "python -c \"import requests; requests.get('http://localhost:8000/health')\""],
        "interval": 30,
        "timeout": 10,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ],
  "volumes": [
    {
      "name": "vectorstore",
      "efsVolumeConfiguration": {
        "fileSystemId": "EFS_FILE_SYSTEM_ID",
        "rootDirectory": "/vectorstore"
      }
    }
  ]
}
```

### Step 6: Create ALB and ECS Service

```bash
# Create target group
TG_ARN=$(aws elbv2 create-target-group \
  --name rag-chatbot-api-tg \
  --protocol HTTP \
  --port 8000 \
  --vpc-id vpc-xxxxxxxx \
  --target-type ip \
  --health-check-path /health \
  --query TargetGroups[0].TargetGroupArn \
  --output text)

# Create ALB
ALB_ARN=$(aws elbv2 create-load-balancer \
  --name rag-chatbot-alb \
  --subnets subnet-xxx subnet-yyy \
  --security-groups sg-xxxxxxxx \
  --query LoadBalancers[0].LoadBalancerArn \
  --output text)

# Create listener
aws elbv2 create-listener \
  --load-balancer-arn $ALB_ARN \
  --protocol HTTP --port 80 \
  --default-actions Type=forward,TargetGroupArn=$TG_ARN

# Create ECS service
aws ecs create-service \
  --cluster rag-chatbot-cluster \
  --service-name rag-chatbot-api \
  --task-definition rag-chatbot \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=$TG_ARN,containerName=rag-api,containerPort=8000"
```

---

## Option 3: AWS Lambda + API Gateway (Serverless)

> Best for low-traffic or cost-sensitive workloads. Note: cold starts can be 5–10s.

```bash
# Install Mangum (ASGI adapter for Lambda)
pip install mangum

# In app/main.py, add at the bottom:
# from mangum import Mangum
# handler = Mangum(app)

# Package with Lambda container image (recommended for size)
# Use the Dockerfile with Lambda base image:
# FROM public.ecr.aws/lambda/python:3.11
```

---

## Option 4: S3 for PDF Storage (Production)

Store PDFs in S3 instead of the container for persistence and sharing:

```bash
# Create S3 bucket
aws s3 mb s3://your-rag-chatbot-pdfs --region us-east-1

# Upload PDFs
aws s3 sync data/pdfs/ s3://your-rag-chatbot-pdfs/pdfs/

# In config.py, add S3 download logic:
# boto3.client('s3').download_file(bucket, key, local_path)
```

---

## Auto-scaling (ECS)

```bash
# Register scalable target
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/rag-chatbot-cluster/rag-chatbot-api \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 1 \
  --max-capacity 10

# Scale on CPU utilisation
aws application-autoscaling put-scaling-policy \
  --policy-name rag-cpu-scaling \
  --service-namespace ecs \
  --resource-id service/rag-chatbot-cluster/rag-chatbot-api \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration \
    '{"TargetValue": 70.0, "PredefinedMetricSpecification": {"PredefinedMetricType": "ECSServiceAverageCPUUtilization"}}'
```

---

## Monitoring & Observability

```bash
# CloudWatch dashboard
aws cloudwatch put-dashboard \
  --dashboard-name RAG-Chatbot \
  --dashboard-body file://aws/cloudwatch-dashboard.json

# Set up alarms
aws cloudwatch put-metric-alarm \
  --alarm-name RAG-API-High-Latency \
  --metric-name TargetResponseTime \
  --namespace AWS/ApplicationELB \
  --statistic Average \
  --period 60 \
  --threshold 2.0 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 3 \
  --alarm-actions arn:aws:sns:us-east-1:ACCOUNT:alerts
```

---

## Cost Estimates

| Component          | Instance       | Monthly Cost (est.) |
|--------------------|----------------|---------------------|
| EC2 t3.medium      | 1 instance     | ~$30                |
| ECS Fargate        | 1 vCPU, 2 GB   | ~$35                |
| ALB                | —              | ~$20                |
| EFS                | 5 GB           | ~$1.50              |
| S3                 | 1 GB PDFs      | ~$0.02              |
| CloudWatch Logs    | 5 GB           | ~$2.50              |
| **Total (ECS)**    |                | **~$60/month**      |

> OpenAI API costs are separate (~$0.002/1K tokens for GPT-4o).

---

## CI/CD with GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to ECS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Login to ECR
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build and push image
        run: |
          docker build -f docker/Dockerfile -t $ECR_URI/rag-chatbot-api:$GITHUB_SHA .
          docker push $ECR_URI/rag-chatbot-api:$GITHUB_SHA

      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster rag-chatbot-cluster \
            --service rag-chatbot-api \
            --force-new-deployment
```
