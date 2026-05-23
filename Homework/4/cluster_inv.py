import numpy as np
from sklearn.cluster import KMeans
import struct
import os

data_path = "/anndata/"
output_dir = "files/"

def load_data(path, dtype):
    with open(path, 'rb') as fin:
        n = struct.unpack('I', fin.read(4))[0]
        d = struct.unpack('I', fin.read(4))[0]

        total_bytes = n * d * np.dtype(dtype).itemsize
        raw_data = fin.read(total_bytes)
        
        data = np.frombuffer(raw_data, dtype=dtype).reshape(n, d)

    return data, n, d

def cluster():
    nlist = 1024
    
    data, n, d = load_data(data_path + "DEEP100K.base.100k.fbin", np.float32)
    kmeans = KMeans(n_clusters=nlist, random_state=0, n_init="auto").fit(data)

    codebook = kmeans.cluster_centers_.astype(np.float32)
    base = kmeans.labels_.astype(np.uint32)

    t_cb = (nlist // 4) * 4
    if t_cb > 0:
        codebook[:t_cb] = codebook[:t_cb].reshape(-1, 4, d).transpose(0, 2, 1).reshape(t_cb, d)

    with open(output_dir + "ivf_codebook.bin", 'wb') as f:
        f.write(struct.pack('I', nlist))
        f.write(struct.pack('I', d))
        f.write(codebook.tobytes())

    offset = np.zeros(nlist + 1, dtype=np.uint32)
    ivflist = np.zeros(n, dtype=np.uint32)
    base2 = np.zeros((n, d), dtype=np.float32)

    invlists = [[] for _ in range(nlist)]
    for idx, label in enumerate(base):
        invlists[label].append(idx)

    temp = 0
    for i in range(nlist):
        offset[i] = temp
        lst = invlists[i]
        length = len(lst)
        
        if length > 0:
            ivflist[temp : temp + length] = lst
            cluster = data[lst]
            t = (length // 4) * 4
            if t > 0:
                cluster[:t] = cluster[:t].reshape(-1, 4, d).transpose(0, 2, 1).reshape(t, d)
            base2[temp : temp + length] = cluster
            
        temp += length
        
    offset[nlist] = temp 

    offset.tofile("files/ivf_offset.bin")
    ivflist.tofile("files/ivf_ivflist.bin")
    base2.tofile("files/ivf_base.bin")

if __name__ == "__main__":
    cluster()