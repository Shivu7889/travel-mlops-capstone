pipeline {
    agent any

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Run Tests') {
            steps {
                sh '''
                    python3 -m venv .jenkins-venv
                    . .jenkins-venv/bin/activate

                    pip install --upgrade pip
                    pip install -r requirements.txt

                    python -m pytest tests -v
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                    docker build \
                      -f docker/Dockerfile \
                      -t travel-mlops-api:latest \
                      .
                '''
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                sh '''
                    kubectl apply -f k8s/deployment.yaml
                    kubectl apply -f k8s/service.yaml
                '''
            }
        }

        stage('Verify Deployment') {
            steps {
                sh '''
                    kubectl rollout status deployment/travel-mlops-api --timeout=120s
                    kubectl get pods
                    kubectl get service travel-mlops-api
                '''
            }
        }
    }

    post {
        success {
            echo 'Travel MLOps CI/CD pipeline completed successfully!'
        }

        failure {
            echo 'Travel MLOps CI/CD pipeline failed.'
        }
    }
}