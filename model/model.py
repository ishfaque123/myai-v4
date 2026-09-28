import numpy as np

class Linear:
    def __init__(self, in_dim, out_dim, rng):
        self.W = rng.normal(0, 0.02, (in_dim, out_dim))
        self.b = np.zeros(out_dim)
    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

class LayerNorm:
    def __init__(self, dim):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
    def forward(self, x):
        self.x = x
        self.mean = x.mean(axis=-1, keepdims=True)
        self.var = x.var(axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + 1e-5)
        self.xhat = (x - self.mean) / self.std
        return self.gamma * self.xhat + self.beta
    def backward(self, dout):
        D = self.x.shape[-1]
        self.dgamma = (dout * self.xhat).sum(axis=0)
        self.dbeta = dout.sum(axis=0)
        dxhat = dout * self.gamma
        dvar = np.sum(dxhat * (self.x - self.mean) * -0.5 * self.std**-3, axis=-1, keepdims=True)
        dmean = np.sum(dxhat * -1/self.std, axis=-1, keepdims=True) + dvar * np.mean(-2*(self.x-self.mean), axis=-1, keepdims=True)
        dx = dxhat/self.std + dvar*2*(self.x-self.mean)/D + dmean/D
        return dx

class MultiHeadAttention:
    def __init__(self, d_model, n_heads, rng):
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.Wq = Linear(d_model, d_model, rng)
        self.Wk = Linear(d_model, d_model, rng)
        self.Wv = Linear(d_model, d_model, rng)
        self.Wo = Linear(d_model, d_model, rng)
    def forward(self, x):
        T, D = x.shape
        H, dh = self.n_heads, self.d_head
        Q = self.Wq.forward(x); K = self.Wk.forward(x); V = self.Wv.forward(x)
        Qh = Q.reshape(T, H, dh).transpose(1,0,2)
        Kh = K.reshape(T, H, dh).transpose(1,0,2)
        Vh = V.reshape(T, H, dh).transpose(1,0,2)
        scores = Qh @ Kh.transpose(0,2,1) / np.sqrt(dh)
        mask = np.triu(np.ones((T,T)), k=1).astype(bool)
        scores = np.where(mask, -1e9, scores)
        scores = scores - scores.max(axis=-1, keepdims=True)
        exp = np.exp(scores)
        attn = exp / exp.sum(axis=-1, keepdims=True)
        out = (attn @ Vh).transpose(1,0,2).reshape(T, D)
        self.cache = (Qh, Kh, Vh, attn, mask, T, D)
        return self.Wo.forward(out)
    def backward(self, dout):
        Qh, Kh, Vh, attn, mask, T, D = self.cache
        H, dh = self.n_heads, self.d_head
        dout_heads = self.Wo.backward(dout).reshape(T, H, dh).transpose(1,0,2)
        dattn = dout_heads @ Vh.transpose(0,2,1)
        dVh = attn.transpose(0,2,1) @ dout_heads
        dscores = attn * (dattn - (dattn*attn).sum(axis=-1, keepdims=True))
        dscores = np.where(mask, 0, dscores) / np.sqrt(dh)
        dQh = dscores @ Kh
        dKh = dscores.transpose(0,2,1) @ Qh
        dQ = dQh.transpose(1,0,2).reshape(T, D)
        dK = dKh.transpose(1,0,2).reshape(T, D)
        dV = dVh.transpose(1,0,2).reshape(T, D)
        return self.Wq.backward(dQ) + self.Wk.backward(dK) + self.Wv.backward(dV)

class FeedForward:
    def __init__(self, d_model, d_ff, rng):
        self.fc1 = Linear(d_model, d_ff, rng)
        self.fc2 = Linear(d_ff, d_model, rng)
    def forward(self, x):
        self.h = self.fc1.forward(x)
        self.mask = self.h > 0
        self.a = self.h * self.mask
        return self.fc2.forward(self.a)
    def backward(self, dout):
        da = self.fc2.backward(dout)
        dh = da * self.mask
        return self.fc1.backward(dh)

class Block:
    def __init__(self, d_model, n_heads, d_ff, rng):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, rng)
        self.ln2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, rng)
    def forward(self, x):
        a = self.attn.forward(self.ln1.forward(x))
        x = x + a
        f = self.ff.forward(self.ln2.forward(x))
        x = x + f
        return x
    def backward(self, dout):
        dx1 = dout + self.ln2.backward(self.ff.backward(dout))
        dx = dx1 + self.ln1.backward(self.attn.backward(dx1))
        return dx

class TinyTransformer:
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, d_ff=256, context_size=32, seed=42):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.context_size = context_size
        self.token_emb = rng.normal(0, 0.02, (vocab_size, d_model))
        self.pos_emb = rng.normal(0, 0.02, (context_size, d_model))
        self.blocks = [Block(d_model, n_heads, d_ff, rng) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size, rng)

    def forward(self, token_ids):
        T = len(token_ids)
        x = self.token_emb[token_ids] + self.pos_emb[:T]
        self._ids, self._T = token_ids, T
        for b in self.blocks:
            x = b.forward(x)
        x = self.ln_f.forward(x)
        return self.head.forward(x)

    def backward(self, dlogits):
        dx = self.head.backward(dlogits)
        dx = self.ln_f.backward(dx)
        for b in reversed(self.blocks):
            dx = b.backward(dx)
        self.dtoken_emb = np.zeros_like(self.token_emb)
        for i, tid in enumerate(self._ids):
            self.dtoken_emb[tid] += dx[i]
        self.dpos_emb = np.zeros_like(self.pos_emb)
        self.dpos_emb[:self._T] = dx

    def _pg(self):
        pg = [(self.token_emb, self.dtoken_emb), (self.pos_emb, self.dpos_emb)]
        for b in self.blocks:
            pg += [(b.ln1.gamma, b.ln1.dgamma), (b.ln1.beta, b.ln1.dbeta),
                   (b.attn.Wq.W, b.attn.Wq.dW), (b.attn.Wq.b, b.attn.Wq.db),
                   (b.attn.Wk.W, b.attn.Wk.dW), (b.attn.Wk.b, b.attn.Wk.db),
                   (b.attn.Wv.W, b.attn.Wv.dW), (b.attn.Wv.b, b.attn.Wv.db),
                   (b.attn.Wo.W, b.attn.Wo.dW), (b.attn.Wo.b, b.attn.Wo.db),
                   (b.ln2.gamma, b.ln2.dgamma), (b.ln2.beta, b.ln2.dbeta),
                   (b.ff.fc1.W, b.ff.fc1.dW), (b.ff.fc1.b, b.ff.fc1.db),
                   (b.ff.fc2.W, b.ff.fc2.dW), (b.ff.fc2.b, b.ff.fc2.db)]
        pg += [(self.ln_f.gamma, self.ln_f.dgamma), (self.ln_f.beta, self.ln_f.dbeta),
               (self.head.W, self.head.dW), (self.head.b, self.head.db)]
        return pg

    def step(self, lr):
        for p, g in self._pg():
            np.clip(g, -5, 5, out=g)
            p -= lr * g

    def save(self, path):
        d = {"token_emb": self.token_emb, "pos_emb": self.pos_emb}
        for i, b in enumerate(self.blocks):
            d[f"b{i}_ln1g"]=b.ln1.gamma; d[f"b{i}_ln1b"]=b.ln1.beta
            d[f"b{i}_wq"]=b.attn.Wq.W; d[f"b{i}_wqb"]=b.attn.Wq.b
            d[f"b{i}_wk"]=b.attn.Wk.W; d[f"b{i}_wkb"]=b.attn.Wk.b
            d[f"b{i}_wv"]=b.attn.Wv.W; d[f"b{i}_wvb"]=b.attn.Wv.b
            d[f"b{i}_wo"]=b.attn.Wo.W; d[f"b{i}_wob"]=b.attn.Wo.b
            d[f"b{i}_ln2g"]=b.ln2.gamma; d[f"b{i}_ln2b"]=b.ln2.beta
            d[f"b{i}_fc1"]=b.ff.fc1.W; d[f"b{i}_fc1b"]=b.ff.fc1.b
            d[f"b{i}_fc2"]=b.ff.fc2.W; d[f"b{i}_fc2b"]=b.ff.fc2.b
        d["lnfg"]=self.ln_f.gamma; d["lnfb"]=self.ln_f.beta
        d["headw"]=self.head.W; d["headb"]=self.head.b
        d["context_size"]=self.context_size
        np.savez(path, **d)

    def load(self, path):
        f = np.load(path)
        self.token_emb = f["token_emb"]; self.pos_emb = f["pos_emb"]
        for i, b in enumerate(self.blocks):
            b.ln1.gamma=f[f"b{i}_ln1g"]; b.ln1.beta=f[f"b{i}_ln1b"]
            b.attn.Wq.W=f[f"b{i}_wq"]; b.attn.Wq.b=f[f"b{i}_wqb"]
            b.attn.Wk.W=f[f"b{i}_wk"]; b.attn.Wk.b=f[f"b{i}_wkb"]
            b.attn.Wv.W=f[f"b{i}_wv"]; b.attn.Wv.b=f[f"b{i}_wvb"]
            b.attn.Wo.W=f[f"b{i}_wo"]; b.attn.Wo.b=f[f"b{i}_wob"]
            b.ln2.gamma=f[f"b{i}_ln2g"]; b.ln2.beta=f[f"b{i}_ln2b"]
            b.ff.fc1.W=f[f"b{i}_fc1"]; b.ff.fc1.b=f[f"b{i}_fc1b"]
            b.ff.fc2.W=f[f"b{i}_fc2"]; b.ff.fc2.b=f[f"b{i}_fc2b"]
        self.ln_f.gamma=f["lnfg"]; self.ln_f.beta=f["lnfb"]
        self.head.W=f["headw"]; self.head.b=f["headb"]

if __name__ == "__main__":
    print("model.py loaded successfully.")
