# Download the pre-trained networks network 
wget --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1B0EeVlHbYBki__WszkbY8A3Za941K8QI' -O checkpoints/models.zip
echo "Models downloaded. Starting to unzip"
unzip -q checkpoints/models.zip -d checkpoints/
rm checkpoints/models.zip
echo "Models downloaded and unzipped."
