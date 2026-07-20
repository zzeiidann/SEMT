# Perhitungan End-to-End: BERT → Autoencoder → Deep Embedded Clustering (DEC)

> **Catatan penting:** Seluruh angka numerik di dokumen ini adalah **contoh ilustrasi**
> untuk menjelaskan mekanisme perhitungan, bukan hasil eksperimen nyata dari dataset kamu.
> Rumus matematisnya adalah rumus asli DEC/BERT sesuai literatur dan kode `SEMTGPU`.
> Semua contoh angka di bawah **saling konsisten** (dipakai dari tahap ke tahap secara berurutan)
> supaya alurnya masuk akal secara matematis.

**Skenario yang dipakai di seluruh dokumen:**
- Jumlah dokumen: $N = 50.000$
- Dimensi output BERT: $768$
- Dimensi laten setelah autoencoder: $256$
- Jumlah klaster/topik: $K = 5$
- Fokus perhitungan: dokumen ke-1, kalimat contoh *"pelayanan restoran ini sangat mengecewakan"*

---

## 1. Tahap BERT — Representasi Teks

### 1.1 Definisi
BERT memetakan satu kalimat input $x$ menjadi vektor representasi $f \in \mathbb{R}^{768}$, diambil dari token `[CLS]` pada layer terakhir encoder.

$$f = H^{(12)}_{[0,:]} \in \mathbb{R}^{768}$$

### 1.2 Perhitungan (dokumen ke-1)

| Tahap | Hasil |
|---|---|
| Tokenisasi | `[CLS], pelayanan, restoran, ini, sangat, mengecewakan, [SEP]` |
| Panjang setelah padding ($L=10$) | 10 token (3 token `[PAD]`) |
| Attention mask | $[1,1,1,1,1,1,1,0,0,0]$ |
| Output akhir (vektor `[CLS]`) | $f_1 \in \mathbb{R}^{768}$ |

Contoh isi vektor $f_1$ (768 dimensi, ditampilkan sebagian):

$$f_1 = [\,0.31,\ -0.42,\ 0.88,\ 0.05,\ \dots,\ -0.19,\ 0.44\,] \in \mathbb{R}^{768}$$

Vektor inilah yang menjadi **input** ke tahap autoencoder berikutnya. Proses ini diulang untuk seluruh $N=50.000$ dokumen sehingga terbentuk matriks:

$$F \in \mathbb{R}^{50000 \times 768}$$

---

## 2. Tahap Autoencoder — Reduksi Dimensi ke Ruang Laten

### 2.1 Definisi
Autoencoder terdiri dari encoder $f_\theta$ dan decoder $g_\varphi$:

$$z = f_\theta(f), \qquad \hat{f} = g_\varphi(z), \qquad z \in \mathbb{R}^{256}$$

Dilatih dengan meminimalkan **Mean Squared Error (MSE)** antara input asli dan hasil rekonstruksi:

$$\mathcal{L}_{MSE} = \frac{1}{N}\sum_{i=1}^{N} \left\| f_i - \hat{f}_i \right\|^2$$

Setelah pretraining, **hanya encoder $f_\theta$ yang dipakai** (decoder dibuang). Encoder ini disebut *shared encoder* karena hasilnya dipakai bersama oleh clustering head dan sentiment head.

### 2.2 Perhitungan (dokumen ke-1)

| Simbol | Dimensi | Keterangan |
|---|---|---|
| $f_1$ | $\mathbb{R}^{768}$ | Input dari BERT |
| $z_1 = f_\theta(f_1)$ | $\mathbb{R}^{256}$ | Output encoder (representasi laten) |

$$z_1 = [\,1.02,\ -0.88,\ 0.55,\ 0.31,\ \dots,\ -0.14,\ 0.27\,] \in \mathbb{R}^{256}$$

Untuk seluruh dataset:

$$Z = \{z_1, z_2, \dots, z_{50000}\} \in \mathbb{R}^{50000 \times 256}$$

Matriks $Z$ inilah **input tunggal** untuk seluruh tahap DEC berikutnya (Bagian 3–7).

---

## 3. Tahap DEC — Inisialisasi Centroid

### 3.1 Definisi

$$\{\mu_j\}_{j=1}^{K} = \text{KMeans}(\{z_i\}_{i=1}^{N},\ K)$$

Di implementasi (`sklearn.cluster.KMeans`), inisialisasi memakai **k-means++** (bukan random murni) dengan `n_init=20` — dijalankan 20 kali percobaan, dipilih hasil dengan inertia (total jarak kuadrat) terkecil.

### 3.2 Perhitungan

Untuk $K=5$, hasil akhir berupa 5 vektor centroid, masing-masing 256 dimensi:

| Centroid | Contoh isi (256-dim, dipotong) |
|---|---|
| $\mu_1$ | $[1.05,\ -0.90,\ 0.52,\ \dots,\ -0.10,\ 0.30]$ |
| $\mu_2$ | $[4.80,\ 5.10,\ -2.20,\ \dots,\ 1.15,\ -0.60]$ |
| $\mu_3$ | $[-3.10,\ 2.05,\ 0.90,\ \dots,\ -1.40,\ 2.10]$ |
| $\mu_4$ | $[0.20,\ -4.50,\ 3.30,\ \dots,\ 0.85,\ -2.05]$ |
| $\mu_5$ | $[2.60,\ 1.15,\ -3.80,\ \dots,\ 0.40,\ 1.90]$ |

$$M = \{\mu_1,\dots,\mu_5\} \in \mathbb{R}^{5 \times 256}$$

Perhatikan $\mu_1$ sengaja dibuat **paling dekat** dengan $z_1$ — ini akan konsisten dipakai di tahap SoftAssign berikutnya (dokumen 1 memang paling cocok ke klaster 1).

---

## 4. Tahap DEC — Soft Assignment (Q)

### 4.1 Definisi

Jarak Euclidean kuadrat di ruang 256 dimensi:

$$\|z_i - \mu_j\|^2 = \sum_{k=1}^{256} (z_{i,k} - \mu_{j,k})^2$$

Soft assignment (distribusi Student's-t, derajat kebebasan 1):

$$q_{ij} = \frac{\left(1+\|z_i-\mu_j\|^2\right)^{-1}}{\sum_{j'=1}^{K}\left(1+\|z_i-\mu_{j'}\|^2\right)^{-1}}$$

### 4.2 Perhitungan (dokumen ke-1, terhadap 5 centroid)

| $j$ | $\|z_1-\mu_j\|^2$ | $(1+\|z_1-\mu_j\|^2)^{-1}$ | $q_{1,j}$ |
|---|---|---|---|
| 1 | 0.62 | 0.6173 | **0.586** |
| 2 | 3.10 | 0.2439 | 0.231 |
| 3 | 8.40 | 0.1064 | 0.101 |
| 4 | 14.75 | 0.0635 | 0.060 |
| 5 | 24.80 | 0.0388 | 0.037 |
| **Total** | — | 1.0699 | **1.000** |

Perhitungan $q_{1,1}$:

$$q_{1,1} = \frac{0.6173}{0.6173+0.2439+0.1064+0.0635+0.0388} = \frac{0.6173}{1.0699} = 0.586$$

$$q_{1,\cdot} = [0.586,\ 0.231,\ 0.101,\ 0.060,\ 0.037]$$

Dilakukan untuk semua 50.000 dokumen → matriks:

$$Q \in \mathbb{R}^{50000 \times 5}$$

---

## 5. Tahap DEC — Distribusi Target (P)

### 5.1 Definisi

Frekuensi soft per klaster (dijumlahkan dari seluruh $N$ dokumen):

$$f_j = \sum_{i=1}^{N} q_{ij}$$

Distribusi target (lebih tajam dari $Q$):

$$p_{ij} = \frac{q_{ij}^2 / f_j}{\sum_{j'=1}^{K} q_{ij'}^2 / f_{j'}}$$

Sifat penting: $\sum_j f_j = N$ selalu (karena tiap baris $Q$ berjumlah 1), dan pembagian dengan $f_j$ mencegah klaster besar mendominasi.

### 5.2 Perhitungan — $f_j$ (agregat dari 50.000 dokumen)

*(nilai ilustratif, hasil penjumlahan kolom Q dari seluruh dataset)*

| $j$ | $f_j$ | Proporsi dokumen ($f_j/N$) |
|---|---|---|
| 1 | 14.500 | 29,0% |
| 2 | 11.000 | 22,0% |
| 3 | 9.800 | 19,6% |
| 4 | 8.200 | 16,4% |
| 5 | 6.500 | 13,0% |
| **Total** | **50.000** | **100%** |

### 5.3 Perhitungan — $p_{1,j}$ (dokumen ke-1)

| $j$ | $q_{1,j}$ | $q_{1,j}^2$ | $f_j$ | $q_{1,j}^2/f_j$ |
|---|---|---|---|---|
| 1 | 0.586 | 0.3434 | 14.500 | $2.368\times10^{-5}$ |
| 2 | 0.231 | 0.0534 | 11.000 | $4.851\times10^{-6}$ |
| 3 | 0.101 | 0.0102 | 9.800 | $1.041\times10^{-6}$ |
| 4 | 0.060 | 0.0036 | 8.200 | $4.390\times10^{-7}$ |
| 5 | 0.037 | 0.0014 | 6.500 | $2.154\times10^{-7}$ |
| **Total** | — | — | — | $\mathbf{3.023\times10^{-5}}$ |

Normalisasi menjadi $p_{1,j}$:

| $j$ | $q_{1,j}^2/f_j$ | $\div$ Total | $p_{1,j}$ |
|---|---|---|---|
| 1 | $2.368\times10^{-5}$ | / $3.023\times10^{-5}$ | **0.783** |
| 2 | $4.851\times10^{-6}$ | / $3.023\times10^{-5}$ | 0.160 |
| 3 | $1.041\times10^{-6}$ | / $3.023\times10^{-5}$ | 0.034 |
| 4 | $4.390\times10^{-7}$ | / $3.023\times10^{-5}$ | 0.015 |
| 5 | $2.154\times10^{-7}$ | / $3.023\times10^{-5}$ | 0.007 |
| **Total** | — | — | **1.000** |

### 5.4 Perbandingan $Q$ vs $P$ (dokumen ke-1)

| $j$ | $q_{1,j}$ (soft assign) | $p_{1,j}$ (target) | Perubahan |
|---|---|---|---|
| 1 | 0.586 | **0.783** | ↑ dipertegas |
| 2 | 0.231 | 0.160 | ↓ ditekan |
| 3 | 0.101 | 0.034 | ↓ ditekan |
| 4 | 0.060 | 0.015 | ↓ ditekan |
| 5 | 0.037 | 0.007 | ↓ ditekan |

**Interpretasi:** $P$ membuat keyakinan model terhadap klaster dominan (klaster 1) makin kuat, sementara klaster lain makin ditekan. $P$ inilah yang menjadi "arah" pembelajaran encoder di tahap optimasi.

---

## 6. Tahap DEC — Optimasi (KL Divergence)

### 6.1 Definisi

$$\mathcal{L}_{KLD}(P,Q) = \sum_{i=1}^{N}\sum_{j=1}^{K} p_{ij}\log\frac{p_{ij}}{q_{ij}}$$

$$\theta,\ \{\mu_j\}^{baru} = \text{Optimize}(P,Q)$$

Karena $z_i = f_\theta(f_i)$, pembaruan $\theta$ (parameter encoder) menyebabkan $z_i$ ikut berubah pada iterasi berikutnya.

### 6.2 Perhitungan kontribusi loss dokumen ke-1

$$\mathcal{L}_{KLD,1} = \sum_{j=1}^{5} p_{1,j}\log\frac{p_{1,j}}{q_{1,j}}$$

| $j$ | $p_{1,j}$ | $q_{1,j}$ | $p_{1,j}/q_{1,j}$ | $\log(\cdot)$ | $p_{1,j}\log(\cdot)$ |
|---|---|---|---|---|---|
| 1 | 0.783 | 0.586 | 1.336 | 0.290 | 0.227 |
| 2 | 0.160 | 0.231 | 0.693 | -0.367 | -0.059 |
| 3 | 0.034 | 0.101 | 0.337 | -1.088 | -0.037 |
| 4 | 0.015 | 0.060 | 0.250 | -1.386 | -0.021 |
| 5 | 0.007 | 0.037 | 0.189 | -1.665 | -0.012 |
| **Total ($\mathcal{L}_{KLD,1}$)** | | | | | **0.098** |

Loss total untuk 1 batch/dataset adalah rata-rata (atau jumlah, tergantung reduksi) dari kontribusi seperti ini di semua 50.000 dokumen:

$$\mathcal{L}_{KLD} = \frac{1}{N}\sum_{i=1}^{N} \mathcal{L}_{KLD,i}$$

Nilai loss ini kemudian di-backpropagate untuk memperbarui bobot encoder $\theta$ dan posisi centroid $\mu_j$ melalui optimizer (Adam, dsb).

---

## 7. Tahap DEC — Kriteria Konvergensi

### 7.1 Definisi

Label klaster hasil hard-assignment:

$$y_{cluster,i} = \arg\max_j q_{ij}$$

Proporsi perubahan label dibanding iterasi sebelumnya:

$$\Delta_{label} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[y_{prev,i} \neq y_{cluster,i}]$$

### 7.2 Perhitungan (untuk dokumen ke-1, lalu digeneralisasi)

Dari tabel $Q$ di Bagian 4, $\arg\max_j q_{1,j} = 1$ (karena $q_{1,1}=0.586$ paling besar) → $y_{cluster,1} = 1$.

Untuk seluruh dataset (contoh agregat):

| Iterasi | Jumlah dokumen berubah label | $\Delta_{label}$ | Status ($\tau = 0{,}001$) |
|---|---|---|---|
| ke-1 | 4.000 / 50.000 | 0,0800 | Belum konvergen |
| ke-2 | 1.500 / 50.000 | 0,0300 | Belum konvergen |
| ke-3 | 210 / 50.000 | 0,0042 | Belum konvergen |
| ke-4 | 35 / 50.000 | 0,0007 | **Konvergen** ✓ |

Ketika $\Delta_{label} < \tau$, iterasi SoftAssign → Target → Optimize dihentikan; posisi centroid dan parameter encoder $\theta$ dianggap stabil.

---

## 8. Ringkasan Alur Dimensi (untuk cepat dicek di slide)

| Tahap | Simbol | Dimensi |
|---|---|---|
| Output BERT | $f_i$ | $\mathbb{R}^{768}$ |
| Semua dokumen (BERT) | $F$ | $\mathbb{R}^{50000\times768}$ |
| Output encoder (laten) | $z_i$ | $\mathbb{R}^{256}$ |
| Semua dokumen (laten) | $Z$ | $\mathbb{R}^{50000\times256}$ |
| Centroid | $\mu_j$ | $\mathbb{R}^{256}$ |
| Semua centroid | $M$ | $\mathbb{R}^{5\times256}$ |
| Soft assignment | $Q$ | $\mathbb{R}^{50000\times5}$ |
| Distribusi target | $P$ | $\mathbb{R}^{50000\times5}$ |
| Loss KLD | $\mathcal{L}_{KLD}$ | skalar (1 angka) |
| $\Delta_{label}$ | — | skalar (1 angka) |

**Poin kunci:** vektor 256-dimensi ($z_i$, $\mu_j$) hanya muncul di representasi mentah. Begitu masuk ke rumus jarak/similarity (SoftAssign, Target, KLD), hasilnya selalu **skalar** — sehingga tabel-tabel perhitungan (Q, P, loss) tetap ringkas meskipun ruang laten aslinya berdimensi tinggi.

---

## 9. Diagram Alur Lengkap

```
Teks input
    │  (tokenisasi, embedding, 12-layer encoder BERT)
    ▼
f ∈ R^768                              ← Bagian 1
    │  (autoencoder: f_θ)
    ▼
z ∈ R^256                              ← Bagian 2
    │
    ├──────────────► KMeans (k-means++, n_init=20)
    │                        │
    │                        ▼
    │                 μ_j ∈ R^256, j=1..5   ← Bagian 3
    │                        │
    ▼                        │
SoftAssign(z, μ) ◄────────────┘
    │
    ▼
Q ∈ R^(N×5)                            ← Bagian 4
    │  (Target: kuadratkan + normalisasi f_j)
    ▼
P ∈ R^(N×5)                            ← Bagian 5
    │
    ▼
L_KLD(P,Q) → backprop → update θ, μ    ← Bagian 6
    │
    ▼
Δ_label < τ ?
    │
    ├── belum → ulangi dari SoftAssign
    └── sudah → STOP, klaster final     ← Bagian 7
```
