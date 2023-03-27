# SSH Keygen

1. Use this line to set up a ssh key pair
```bash
ssh-keygen -t rsa -b 4096
```

2. Display the generated public key
```bash
cat ~/.ssh/id_rsa.pub
```

3. Copy the contents

4. Past Content in Vast.ai

5. Once you have set it up you can connect to the machine using the custom ssh command

6. when finished with the model training, we can copy our logs and models to our local machine with the following command
```bash
scp -r -P 10200 root@107.222.215.224:./logs ./Desktop
````

- 10200 is the port and specified by vast
- root is the username of the vm
- 107.222.215.224 is the IP-adress of our vm    


7. log on ssh server how much RAM of GPUs is being used every second
```bash
watch -n 1 nvidia-smi
```