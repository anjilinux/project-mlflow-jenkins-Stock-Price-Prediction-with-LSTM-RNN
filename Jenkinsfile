pipeline {
    agent any
    stages {
        stage('Setup') {
            steps {
                sh 'python3 -m venv venv'
                sh '. venv/bin/activate && pip install -r requirements.txt'
            }
        }
        stage('Test') {
            steps {
                sh '. venv/bin/activate && pytest'
            }
        }
        stage('Train Model') {
            steps {
                sh '. venv/bin/activate && python src/train.py'
            }
        }
        stage('Deploy') {
            steps {
                sh '. venv/bin/activate && python src/app.py &'
            }
        }
    }
}
