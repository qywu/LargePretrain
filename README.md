# LargePretrain

This repo is to train large-scale language models on GCP

## Usage

1. First setup GCP and create instance for the training. You can find detailed instructions [here](https://github.com/qywu/TorchFly/tree/master/examples/GCP%20Preemptible).

    ```bash
    gcloud compute instances create qywu-preemptible \
        --image nvidia-gpu-cloud-image-pytorch-20191120 \
        --image-project nvidia-ngc-public \
        --zone us-west1-b \
        --custom-vm-type n1 \
        --custom-cpu 32 \
        --custom-memory 208GB \
        --boot-disk-size 32GB \
        --boot-disk-type pd-ssd \
        --tags qywu-network-rules \
        --accelerator type=nvidia-tesla-v100,count=8 \
        --scopes cloud-platform \
        --preemptible 
    ```

2. Then SSH to remote host

    ```bash
    gcloud beta compute ssh --zone "us-west1-b" "qywu-preemptible" --project "nlp-compute-project"
    ```

3. Install GCSFuse

    ```bash
    export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
    echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

    sudo apt-get update
    sudo apt-get install gcsfuse
    ```

4. Then exectute the following:

    ```bash
    sudo su
    mkdir /data

    mkdir /workspace
    cd /workspace
    git clone https://github.com/qywu/LargePretrain
    cd /workspace/LargePretrain
    git checkout -f gcp_exp1
    git pull
    chmod -R 777 /workspace
    
    mkdir outputs
    docker build -f docker/Dockerfile -t pretrain .
    ```

5. Add StartUp Script

    ```bash
    gcloud compute instances add-metadata --zone "us-west1-b" qywu-preemptible \
        --metadata-from-file startup-script=startup_script.sh
    ```

6. Check Status

    ```bash
    gcloud beta compute ssh --zone "us-west1-b" "qywu-preemptible" --project "nlp-compute-project"
    ```