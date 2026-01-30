pipeline {
    agent any

    environment {
        MLFLOW_TRACKING_URI = "http://localhost:5555"
        MLFLOW_EXPERIMENT_NAME = "stock7"
    }

    stages {

        stage('Setup Environment') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install -r requirements.txt
                '''
            }
        }

        stage('Data Preprocessing') {
            steps {
                sh '''
                . venv/bin/activate
                python data_preprocessing.py
                '''
            }
        }

        stage(' Model.py-file') {
            steps {
                sh '''
                . venv/bin/activate
                python model.py
                '''
            }
        }       


        // stage('Nuke MLflow State') {
        //     steps {
        //         sh '''
        //         rm -rf /var/lib/jenkins/mlflow_clean
        //         rm -rf /var/lib/jenkins/mlruns
        //         rm -rf ~/.mlflow
        //         rm -rf ~/.cache/mlflow
        //         rm -rf mlruns
        //         mkdir -p /var/lib/jenkins/mlflow_clean
        //         '''
        //     }
        // }



        stage('Train Model') {
            steps {
                sh '''
                . venv/bin/activate
                python train.py
                '''
            }
        }


        stage('Evaluate Model') {
            steps {
                sh '''
                . venv/bin/activate
                python evaluate.py
                '''
            }
        }


        stage('predic-py-file') {
            steps {
                sh '''
                . venv/bin/activate
                python predict.py
                '''
            }
        }

        stage('app.py') {
            steps {
                sh '''
                . venv/bin/activate
                python app.py
                '''
            }
        }


        stage('Test Model') {
            steps {
                sh '''
                . venv/bin/activate
                pytest test_model.py
                '''
            }
        }
    }
}
