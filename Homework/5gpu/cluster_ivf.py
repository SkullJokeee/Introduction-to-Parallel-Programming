import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import struct

data_path = "anndata/"

def load_data(path, dtype):
    with open(path, 'rb') as fin:
        n = struct.unpack('I', fin.read(4))[0]
        d = struct.unpack('I', fin.read(4))[0]
        total_bytes = n * d * np.dtype(dtype).itemsize
        raw_data = fin.read(total_bytes)
        data = np.frombuffer(raw_data, dtype=dtype).reshape(n, d)
    return data, n, d

def save(filepath, data, n, d):
    with open(filepath, 'wb') as f:
        f.write(struct.pack('I', n))
        f.write(struct.pack('I', d))
        f.write(data.tobytes())

def cluster():
    nlist = 1024
    
    data, n, d = load_data(data_path + "DEEP100K.base.100k.fbin", np.float32)
    data = normalize(data, norm='l2', axis=1)
    kmeans = KMeans(n_clusters=nlist, random_state=0, n_init="auto").fit(data)

    codebook = kmeans.cluster_centers_.astype(np.float32)
    base = kmeans.labels_.astype(np.uint32)

    save("files/ivf_codebook.bin", codebook, nlist, d)

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
            base2[temp : temp + length] = data[lst]
            
        temp += length
        
    offset[nlist] = temp 

    save("files/ivf_offset.bin", offset, nlist + 1, 1)
    save("files/ivf_ivflist.bin", ivflist, n, 1)    
    save("files/ivf_base.bin", base2, n, d)

if __name__ == "__main__":
    cluster()