import torchreid

# Ghi đè đường dẫn root để dữ liệu nằm trong thư mục 'dataset'
torchreid.data.register_image_dataset('market1501', torchreid.data.datasets.image.market1501.Market1501)

# Tự động tải và xử lý dataset vào 'dataset/market1501'
torchreid.data.ImageDatasetManager(root='dataset').get('market1501')
