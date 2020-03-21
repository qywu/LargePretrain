# LargePretrain

This repo is to train large-scale language models on GCP

## Usage

1. First setup GCP and create instance for the training. You can find useful instructions [here](https://github.com/qywu/TorchFly/tree/master/examples/GCP%20Preemptible).

2. Then exectute the following:

```bash
sudo su
mkdir /data
git clone https://github.com/qywu/LargePretrain
cd LargePretrain
mkdir outputs
docker build -f docker/Dockerfile -t pretrain .
```

3. Add StartUp Script

```bash
gcloud compute instances add-metadata --zone "us-west1-a" qywu-preemptible \
    --metadata-from-file startup-script=startup_script.sh
```

4. Check Status

```bash
gcloud beta compute ssh --zone "us-west1-a" "qywu-preemptible" --project "nlp-compute-project"
```