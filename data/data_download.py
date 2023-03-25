import gdown

url = "https://drive.google.com/u/1/uc?id=1rt0-DRWWNb18Y2RchhAgx3y2ig2fsB7v&export=download"
output = "test.png"
gdown.download(url, output)