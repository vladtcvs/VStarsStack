pipeline {
    agent any
    stages {
        stage('Dependencies') {
            steps {
                withPythonEnv('python3') {
                    sh 'python3 -m pip install pylint'
                    sh 'python3 -m pip install numpy'
                    sh 'python3 -m pip install opencv-python'
                    sh 'python3 -m pip install imutils'
                    sh 'python3 -m pip install astropy'
                    sh 'python3 -m pip install rawpy'
                    sh 'python3 -m pip install scikit-image'
                    sh 'python3 -m pip install scipy'
                    sh 'python3 -m pip install imutils'
                    sh 'python3 -m pip install exifread'
                    sh 'python3 -m pip install matplotlib'
                    sh 'python3 -m pip install pillow'
                }
            }
        }
        stage('Build & Install') {
            steps {
                withPythonEnv('python3') {
                    sh 'python3 setup.py clean'
                    sh 'python3 setup.py build'
                    sh 'python3 setup.py install'
                }
            }
        }
        stage('Lint') {
            steps {
                withPythonEnv('python3') {
                    sh 'pylint --rcfile=pylint.rc --fail-under=6 src/'
                }
            }
        }
        stage('Tests') {
            steps {
                withPythonEnv('python3') {
                    sh 'vstarstack'
                }
            }
        }
    }
}
