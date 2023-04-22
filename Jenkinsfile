pipeline {
    stages {
        stage("Linting") {
            steps {
                script {
                    sh """
                    pylint --rcfile=pylint.rc src/vstarstack/**/*.py
                    """
                }
            }
        }
    }
}
