# RBM 波動関数による 2 次元横磁場イジング模型 VMC 実装計画

この文書は、2 次元横磁場イジング模型の基底状態エネルギーを Restricted Boltzmann Machine (RBM) 波動関数と変分モンテカルロ (VMC) 法で求める教育用 Python プロジェクトの実装方針をまとめる。現時点のリポジトリは実質的に空であるため、ここで採用する物理・数値上の仮定を明示し、以後の実装が曖昧にならないようにする。

## 基本方針と制約

- 初期実装で使う外部依存は **NumPy** と **pytest** のみとする。
- NetKet、JAX、PyTorch は使わない。
- 小さい格子では厳密対角化 (Exact Diagonalization; ED) と比較し、VMC 実装を検証できるようにする。
- 教育用プロジェクトとして、実装は過度に細かく分割しない。まずは読みやすさとテストしやすさを優先する。
- 不明点や選択肢が複数ある箇所は、この計画書または README に採用した仮定を明記する。
- この計画段階では source code は変更しない。以後の実装段階で Python モジュール、テスト、README を追加する。

## 採用するハミルトニアンと符号規約

2 次元正方格子上の横磁場イジング模型として、以下のハミルトニアンを採用する。

\[
H = -J \sum_{\langle i,j \rangle} \sigma_i^z \sigma_j^z - \Gamma \sum_i \sigma_i^x.
\]

ここで、

- \(\sigma_i^z, \sigma_i^x\) はサイト \(i\) 上の Pauli 演算子。
- スピン配置は \(\sigma_i^z\) の固有値 \(s_i \in \{-1, +1\}\) で表す。
- \(J > 0\) は強磁性的相互作用を表す。
- \(\Gamma \ge 0\) は横磁場の強さを表す。
- エネルギーは全エネルギーとし、必要に応じてサイト数 \(N=L^2\) で割ったエネルギー密度も出力する。

この符号規約では、\(\Gamma=0\), \(J>0\) の古典極限において、全スピンが同じ向きの 2 つの配置が基底状態になる。

## 格子・周期境界条件・ボンドの数え方

格子は \(L \times L\) の 2 次元正方格子とし、サイト数は

\[
N = L^2
\]

である。サイトの 1 次元インデックスは

\[
i = x + L y, \quad x,y \in \{0,1,\ldots,L-1\}
\]

で定義する。

周期境界条件を両方向に課す。各サイト \((x,y)\) から右隣 \((x+1 \bmod L, y)\) と上隣 \((x, y+1 \bmod L)\) へのボンドのみを列挙することで、最近接ボンドを重複なく数える。

- 通常の \(L \ge 2\) ではボンド数は \(2N = 2L^2\)。
- 教育用実装では、まず \(L \ge 2\) をサポート対象とする。
- \(L=1\) は周期境界条件下で自己ボンドの扱いが特殊になるため、初期実装では明示的に非サポートまたは特別扱いとする。

## RBM 波動関数の定義

可視変数はスピン配置

\[
\mathbf{s} = (s_1,\ldots,s_N), \quad s_i \in \{-1,+1\}
\]

とする。隠れユニット数は

\[
M = \lfloor \mathrm{hidden\_density} \times N \rfloor
\]

を基本とし、少なくとも 1 以上になるようにする。

RBM の複素波動関数を

\[
\psi_\theta(\mathbf{s}) = \exp\left(\sum_i a_i s_i\right)
\prod_{k=1}^{M} 2\cosh\left(b_k + \sum_i W_{ik}s_i\right)
\]

で定義する。パラメータは

- 可視バイアス \(a_i\)
- 隠れバイアス \(b_k\)
- 重み \(W_{ik}\)

であり、一般には複素数とする。ただし教育用の最初の段階では、実装とデバッグを簡単にするため実数パラメータから開始してもよい。その場合は「実数 RBM を初期実装とする」という仮定を README または本計画書に追記する。

数値安定性と効率のため、実装では \(\psi\) そのものではなく

\[
\log \psi_\theta(\mathbf{s})
= \sum_i a_i s_i + \sum_{k=1}^{M} \log\left[2\cosh\left(\theta_k(\mathbf{s})\right)\right]
\]

を主に扱う。ここで

\[
\theta_k(\mathbf{s}) = b_k + \sum_i W_{ik}s_i.
\]

## Metropolis 法で更新する対象と、Adam または SR で更新する対象の区別

VMC では 2 種類の更新を明確に区別する。

### Metropolis 法で更新する対象

Metropolis 法では、RBM パラメータ \(\theta\) を固定したまま、確率分布

\[
P_\theta(\mathbf{s}) = \frac{|\psi_\theta(\mathbf{s})|^2}{\sum_{\mathbf{s}'} |\psi_\theta(\mathbf{s}')|^2}
\]

に従うスピン配置 \(\mathbf{s}\) をサンプリングする。

- 更新対象はスピン配置 \(\mathbf{s}\)。
- 提案は 1 スピン反転 \(s_i \to -s_i\) を基本とする。
- 受理確率は

\[
A(\mathbf{s}\to\mathbf{s}') = \min\left(1,
\left|\frac{\psi_\theta(\mathbf{s}')}{\psi_\theta(\mathbf{s})}\right|^2\right)
= \min\left(1, \exp\left[2\operatorname{Re}(\log\psi_\theta(\mathbf{s}')-\log\psi_\theta(\mathbf{s}))\right]\right).
\]

### Adam または SR で更新する対象

Adam または Stochastic Reconfiguration (SR) では、サンプルされたスピン配置を使ってエネルギー期待値を下げるように RBM パラメータ \(\theta=(a,b,W)\) を更新する。

- 更新対象は RBM パラメータ \(a_i, b_k, W_{ik}\)。
- Metropolis サンプリング中に \(\theta\) は固定する。
- 1 回の VMC iteration は「現在の \(\theta\) でサンプリング → 局所エネルギーと対数微分を評価 → optimizer で \(\theta\) を更新」という順序にする。
- 初期実装では Adam を優先し、SR は後続段階の拡張候補にする。SR を実装する場合は、正則化 \(S+\lambda I\) の \(\lambda\) を設定値として README または計画書に明記する。

## 局所エネルギーの式

スピン配置 \(\mathbf{s}\) に対する局所エネルギーは

\[
E_{\mathrm{loc}}(\mathbf{s}) = \frac{(H\psi)(\mathbf{s})}{\psi(\mathbf{s})}
\]

である。採用したハミルトニアンでは、対角項と非対角項に分けて

\[
E_{\mathrm{loc}}(\mathbf{s}) =
-J \sum_{\langle i,j \rangle} s_i s_j
- \Gamma \sum_i \frac{\psi_\theta(\mathbf{s}^{(i)})}{\psi_\theta(\mathbf{s})}
\]

となる。ここで \(\mathbf{s}^{(i)}\) はサイト \(i\) のスピンだけを反転した配置である。

実装では比

\[
\frac{\psi_\theta(\mathbf{s}^{(i)})}{\psi_\theta(\mathbf{s})}
= \exp\left(\log\psi_\theta(\mathbf{s}^{(i)}) - \log\psi_\theta(\mathbf{s})\right)
\]

を用いて計算する。

## 厳密対角化による検証方針

小さい系では全 Hilbert 空間 \(2^N\) を明示的に扱い、厳密対角化で基底状態エネルギーを求める。

- 対象はまず \(L=2\) を標準検証ケースとする。\(N=4\), Hilbert 空間次元は 16。
- 余裕があれば \(L=3\) も検証できるが、\(N=9\), 次元 512 であり、密行列対角化でも教育用途ではまだ扱える範囲とする。
- 初期実装では NumPy の `numpy.linalg.eigh` を使った密行列対角化を採用する。
- ED ハミルトニアンと VMC 局所エネルギーで、同じボンド列挙と同じ符号規約を使う。
- 検証テストでは以下を確認する。
  - \(\Gamma=0\) での古典基底エネルギーが \(-J \times \text{number_of_bonds}\) になること。
  - \(L=2\) の ED エネルギーが固定 seed の VMC 推定値と粗い許容誤差内で整合すること。
  - RBM の局所エネルギー計算が、明示的なハミルトニアン作用から得た値と一致すること。

## 5 段階の実装計画

### Stage 1: プロジェクト骨格と格子ユーティリティ

追加予定ファイル:

- `README.md`
- `pyproject.toml`
- `src/rbm_wave_state/__init__.py`
- `src/rbm_wave_state/lattice.py`
- `tests/test_lattice.py`

実装内容:

- パッケージ構成を作る。
- \(L\times L\) 格子のサイト番号変換を実装する。
- 周期境界条件つき最近接ボンド列挙を実装する。
- \(L \ge 2\) のみをサポートする仮定を明記する。

テスト:

- `L=2`, `L=3` でボンド数が \(2L^2\) になること。
- ボンドが重複していないこと。
- サイト番号と \((x,y)\) の相互変換が一致すること。

実行コマンド:

```bash
python -m pytest tests/test_lattice.py
```

### Stage 2: ハミルトニアンと厳密対角化

追加予定ファイル:

- `src/rbm_wave_state/hamiltonian.py`
- `src/rbm_wave_state/exact.py`
- `tests/test_hamiltonian.py`
- `tests/test_exact.py`

実装内容:

- 対角 Ising エネルギー \(-J\sum_{\langle i,j\rangle}s_is_j\) を実装する。
- 1 スピン反転を通じて横磁場項を含む密行列ハミルトニアンを作る。
- `numpy.linalg.eigh` による基底状態エネルギー計算を実装する。

テスト:

- \(\Gamma=0\) の ED 基底エネルギーが \(-J\times 2L^2\) と一致すること。
- ハミルトニアン行列が Hermitian であること。
- `L=2`, `J=1`, `Gamma=0` の基底状態縮退または最低固有値を確認すること。

実行コマンド:

```bash
python -m pytest tests/test_hamiltonian.py tests/test_exact.py
```

### Stage 3: RBM 波動関数と局所エネルギー

追加予定ファイル:

- `src/rbm_wave_state/rbm.py`
- `src/rbm_wave_state/local_energy.py`
- `tests/test_rbm.py`
- `tests/test_local_energy.py`

実装内容:

- RBM パラメータの初期化を実装する。
- `log_psi(spins)` と 1 スピン反転比を実装する。
- 局所エネルギー \(E_{\mathrm{loc}}(\mathbf{s})\) を実装する。
- 最初は実数パラメータを既定とし、複素数対応は API が複雑にならない範囲で追加する。

テスト:

- `log_psi` が同じ seed で再現すること。
- 反転比が直接計算した `exp(log_psi(flipped)-log_psi(original))` と一致すること。
- 局所エネルギーが ED の明示的なハミルトニアン作用から計算した値と一致すること。

実行コマンド:

```bash
python -m pytest tests/test_rbm.py tests/test_local_energy.py
```

### Stage 4: Metropolis サンプリングと Adam 最適化

追加予定ファイル:

- `src/rbm_wave_state/sampler.py`
- `src/rbm_wave_state/optim.py`
- `src/rbm_wave_state/vmc.py`
- `tests/test_sampler.py`
- `tests/test_optim.py`
- `tests/test_vmc.py`

実装内容:

- 複数 chain の Metropolis サンプラーを実装する。
- warmup、sweeps per sample、sample 数を設定可能にする。
- RBM パラメータの対数微分を実装する。
- エネルギー勾配

  \[
  \partial_{\theta^*} E \approx 2\left(\langle E_{\mathrm{loc}} O_\theta^*\rangle - \langle E_{\mathrm{loc}}\rangle\langle O_\theta^*\rangle\right)
  \]

  を使って Adam 更新を行う。実数パラメータのみの場合は実部を用いる。
- `run_vmc(config)` のような単純な高水準 API を用意する。

テスト:

- 固定 seed でサンプラーの結果が再現すること。
- Metropolis の受理率が 0 から 1 の範囲にあること。
- Adam が単純なダミー勾配でパラメータを期待方向に更新すること。
- 短い VMC run が例外なく実行でき、エネルギー履歴を返すこと。

実行コマンド:

```bash
python -m pytest tests/test_sampler.py tests/test_optim.py tests/test_vmc.py
```

### Stage 5: CLI、検証例、ドキュメント整備

追加予定ファイル:

- `src/rbm_wave_state/cli.py`
- `examples/run_vmc_l2.py`
- `examples/compare_exact_l2.py`
- `tests/test_cli.py`
- README の使用例更新

実装内容:

- `python -m rbm_wave_state.cli` で小さい VMC 実験を実行できるようにする。
- `L=2` で ED と VMC を比較する例を追加する。
- 典型的なパラメータ、出力の読み方、既知の制限を README に記載する。

テスト:

- CLI が `--help` を表示できること。
- 小さいパラメータで CLI が正常終了すること。
- `examples/compare_exact_l2.py` が ED エネルギーと VMC エネルギーを出力すること。

実行コマンド:

```bash
python -m pytest tests/test_cli.py
python examples/compare_exact_l2.py
```

## 想定する数値パラメータ一覧

| パラメータ | 型 | 既定値案 | 説明 |
| --- | --- | ---: | --- |
| `L` | int | `2` | 正方格子の一辺の長さ。初期実装では `L >= 2`。 |
| `J` | float | `1.0` | Ising 相互作用。`J > 0` を強磁性とする。 |
| `Gamma` | float | `1.0` | 横磁場の強さ。`Gamma >= 0` を想定する。 |
| `hidden_density` | float | `2.0` | 隠れユニット数を `M = floor(hidden_density * L * L)` で決める。最低 1。 |
| `n_chains` | int | `16` | 並列に走らせる Metropolis chain 数。 |
| `n_warmup` | int | `100` | サンプル取得前の warmup sweep 数。 |
| `sweeps_per_sample` | int | `1` | サンプル間に行う sweep 数。1 sweep は全サイト数 `N` 回の 1 スピン反転提案。 |
| `n_samples` | int | `1000` | 1 iteration あたりのサンプル数。全 chain 合計として扱う方針。 |
| `n_iterations` | int | `100` | VMC 最適化 iteration 数。 |
| `learning_rate` | float | `0.01` | Adam の学習率。 |
| `seed` | int | `1234` | NumPy 乱数生成器の seed。 |

## 初期実装で明記すべき仮定

- 初期サポートは \(L \ge 2\) の正方格子とする。
- 周期境界条件を使い、右向き・上向きボンドのみを列挙する。
- `n_samples` は全 chain 合計のサンプル数として解釈する。
- 最初の optimizer は Adam とする。SR は後続拡張扱いとする。
- RBM パラメータは実数から開始し、複素数対応はテストと API 方針が固まってから追加する。
- VMC は確率的手法であるため、ED との比較テストでは厳密一致ではなく、十分に粗い許容誤差を用いる。
