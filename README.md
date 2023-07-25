# cup
QC module for Luminex based PRISM data



Running on AWS (Production)
Not Implemented

Running on AWS (Development)
When code is checked into this branch, a jenkins job (https://jenkins.clue.io/view/DEPLOY-DEV/job/DEPLOY-CUP-DEV/) is kicked off; this jenkins job kicks off another process on prism.clue.io (server AWS cloud). The process is a bash script located at /cmap/start-up-scripts/docker_cup_dev.sh (Not yet under version control) The process deploys a docker container which is accessible at https://prism.clue.io/cup-dev/
