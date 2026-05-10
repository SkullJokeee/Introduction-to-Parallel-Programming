import numpy as np
from sklearn.cluster import KMeans
import struct
import numpy as np

data_path = "/anndata/"

def load_data(path, dtype):

    with open(path, 'rb') as fin:
        n = struct.unpack('I', fin.read(4))[0]
        d = struct.unpack('I', fin.read(4))[0]
        
        total_bytes = n * d * np.dtype(dtype).itemsize
        raw_data = fin.read(total_bytes)
        
        data = np.frombuffer(raw_data, dtype=dtype).reshape(n, d)
    
    return data, n, d


def cluster():
    vecdim = 96
    sub_dim = vecdim // 4

    data,n,d = load_data(data_path + "DEEP100K.base.100k.fbin", np.float32)

    codebook = np.zeros((4, 256, sub_dim), dtype=np.float32)
    base = np.zeros((n, 4), dtype=np.uint8)

    for i in range(0,4):
        sub_data = data[:, i*sub_dim : (i+1)*sub_dim]
        
        kmeans = KMeans(n_clusters=256, random_state=0, n_init="auto").fit(sub_data)
        
        # codebook.append(kmeans.cluster_centers_)
        codebook[i] = kmeans.cluster_centers_
        base[:, i] = kmeans.labels_

    with open("files/pq_codebook.bin", 'wb') as f:
        f.write(struct.pack('I', 4 * 256)) 
        f.write(struct.pack('I', sub_dim))         
        f.write(codebook.tobytes())

    with open("files/pq_base.bin", 'wb') as f:
        f.write(struct.pack('I', n))         
        f.write(struct.pack('I', 4)) 
        f.write(base.tobytes())


if __name__ == "__main__":
    cluster()
