import numpy as np
from sklearn.cluster import KMeans
import struct

data_path = "/anndata/"

def load_data(path, dtype):
    with open(path, 'rb') as f:
        n = struct.unpack('I', f.read(4))[0]
        d = struct.unpack('I', f.read(4))[0]
        raw = f.read(n * d * np.dtype(dtype).itemsize)
        data = np.frombuffer(raw, dtype=dtype).reshape(n, d)
    return data, n, d

def save(filepath, data, n, d):
    with open(filepath, 'wb') as f:
        f.write(struct.pack('I', n))
        f.write(struct.pack('I', d))
        f.write(data.tobytes())

def cluster():
    nlist = 128 ###
    m = 4
    k = 256
    data, n, d = load_data(data_path + "DEEP100K.base.100k.fbin", np.float32)
    sub_d = d // m

    ivf_km = KMeans(n_clusters=nlist, random_state=0, n_init="auto").fit(data)
    ivf_cb = ivf_km.cluster_centers_.astype(np.float32)
    labels = ivf_km.labels_.astype(np.uint32)

    t_cb = (nlist // 4) * 4
    if t_cb > 0:
        ivf_cb[:t_cb] = ivf_cb[:t_cb].reshape(-1, 4, d).transpose(0, 2, 1).reshape(t_cb, d)

    save("files/ivfpq_ivf_cb.bin", ivf_cb, nlist, d)

    pq_cbs = np.zeros((nlist, m, k, sub_d), dtype=np.float32)
    offset = np.zeros(nlist + 1, dtype=np.uint32)
    lst = np.zeros(n, dtype=np.uint32)
    base2 = np.zeros((n, m), dtype=np.uint8)

    inv = [[] for _ in range(nlist)]
    for i, l in enumerate(labels):
        inv[l].append(i)

    start = 0
    for i in range(nlist):
        offset[i] = start
        curr = inv[i]
        length = len(curr)
        
        if length > 0:
            lst[start : start + length] = curr
            sub = data[curr]
            pq_base = np.zeros((length, m), dtype=np.uint8)
            
            for j in range(m):
                part = sub[:, j*sub_d : (j+1)*sub_d]
                n_k = min(k, length)
                if n_k > 0:
                    km = KMeans(n_clusters=n_k, random_state=0, n_init="auto").fit(part)
                    pq_cbs[i, j, :n_k] = km.cluster_centers_
                    pq_base[:, j] = km.labels_
            
            base2[start : start + length] = pq_base
            
        start += length
        
    offset[nlist] = start 

    save("files/ivfpq_pq_cb.bin", pq_cbs, nlist * m * k, sub_d)
    save("files/ivfpq_offset.bin", offset, nlist + 1, 1)
    save("files/ivfpq_list.bin", lst, n, 1)
    save("files/ivfpq_base.bin", base2, n, m)

if __name__ == "__main__":
    cluster()