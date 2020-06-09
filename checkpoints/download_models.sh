# Download the pre-trained networks network 
wget --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1iKUTfmYha86slBb6U__BtWCNWiOghs0z' -O checkpoints/models.zip
echo "Models downloaded. Starting to unzip"
unzip -q checkpoints/models.zip -d checkpoints/
rm checkpoints/models.zip
echo "Models downloaded and unzipped."
