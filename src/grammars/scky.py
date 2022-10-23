import torch
from torch_struct.helpers import _Struct, Chart
from torch_struct.distributions import StructDistribution

A, B = 0, 1


class SCKY(_Struct):
    def logpartition(self, scores, lengths=None, force_grad=False):

        semiring = self.semiring

        # Checks
        terms, rules, roots = scores
        rules.requires_grad_(True)
        ssize = semiring.size()
        batch, N_, M_, _, _, T = terms.shape
        _, NT, _, _ = rules.shape
        S = NT + T

        # The inputs should be padded on left and right (bos/eos)
        N = N_ - 2
        M = M_ - 2

        terms, rules, roots = (
            semiring.convert(terms).requires_grad_(True),
            semiring.convert(rules).requires_grad_(True),
            semiring.convert(roots).requires_grad_(True),
        )
        if lengths is None:
            lengths = (
                torch.LongTensor([[N, M]]).expand(batch, 2).to(terms.device)
            )

        # Charts
        beta = [
            Chart((batch, N + 1, M + 1, N + 1, M + 1, NT), rules, semiring)
            for _ in range(2)
        ]
        spans = []
        span_ws = []
        v = (ssize, batch)
        term_use = terms + 0.0
        Y_term_use = term_use[..., 1:, 1:, :, :, :]
        # Y_term_use[b, i, j, :, :, k] = [[00, 01],[10, 11]]
        # rotate so that Z_term_use[b, i, j, :, :, k] = [[11, 10],[01, 00]]
        Z_term_use = torch.rot90(term_use[..., :-1, :-1, :, :, :], 2, [-3, -2])

        # Split into NT/T groups
        NTs = slice(0, NT)
        Ts = slice(NT, S)
        rules = rules.view(ssize, batch, 1, NT, S, S)

        def arr(a, b):
            # extra 1 to broadcast over NxM
            return (
                rules[..., a, b]
                .contiguous()
                .view(*v + (1, NT, -1))
                .transpose(-2, -1)
            )

        matmul = semiring.matmul
        times = semiring.times
        X_Y_Z = arr(NTs, NTs)
        X_Y1_Z = arr(Ts, NTs)
        X_Y_Z1 = arr(NTs, Ts)
        X_Y1_Z1 = arr(Ts, Ts)

        def mask_corners(Y, Z):
            # don't align null/x with y/null
            wn, wm = Y.shape[-3], Y.shape[-2]
            mask = torch.zeros(
                (1, 1, 1, 1, wn, wm, 1), dtype=bool, device=Y.device
            )
            for i, j in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
                mask[..., i, j, :] = True
            return Y.masked_fill(mask, -1e5), Z.masked_fill(mask, -1e5)

        def combine(Y, Z, XYZ):
            # print(Y.shape, Z.shape, XYZ.shape)
            try:
                vy = Y.shape[:-3] + (-1, Y.shape[-1])
                vz = Z.shape[:-3] + (-1, Z.shape[-1])
                v2 = Y.shape[:-3] + (-1,)
                YZ = matmul(
                    torch.reshape(Y, vy).transpose(-2, -1), torch.reshape(Z, vz)
                )
                return matmul(YZ.view(*v2), XYZ)
            except Exception as e:
                print(Y.shape, Z.shape, XYZ.shape)
                raise e

        for wn in range(0, N + 1):
            for wm in range(0, M + 1):
                if wn <= 1 and wm <= 1:
                    continue

                all_span = []

                # NT -> NT NT
                Y_ = beta[A][: N - wn + 1, : M - wm + 1, : wn + 1, : wm + 1]
                Z_ = beta[B][wn:, wm:, N - wn :, M - wm :]
                Y, Z = mask_corners(Y_, Z_)
                X1 = combine(Y, Z, X_Y_Z)
                all_span.append(X1)

                # NT -> T NT
                Y_term = Y_term_use[
                    ..., : N - wn + 1, : M - wm + 1, : wn + 1, : wm + 1, :
                ]
                Z_rest = Z[..., : min(2, wn + 1), : min(2, wn + 1), :]
                X2 = combine(Y_term, Z_rest, X_Y1_Z)
                all_span.append(X2)

                # NT -> NT T
                Z_term = Z_term_use[..., wn:, wm:, -(wn + 1) :, -(wm + 1) :, :]
                Y_rest = Y[..., -min(2, wn + 1) :, -min(2, wm + 1) :, :]
                X3 = combine(Y_rest, Z_term, X_Y_Z1)
                all_span.append(X3)

                # NT -> T T
                if wn <= 2 and wm <= 2:
                    ln = 1 if wn == 2 else None
                    lm = 1 if wm == 2 else None
                    Y_t = Y_term[..., ln:, lm:, :]
                    Z_t = Z_term[..., :ln, :lm, :]
                    X4 = combine(Y_t, Z_t, X_Y1_Z1)
                    all_span.append(X4)

                span = semiring.sum(torch.stack(all_span, dim=-1))
                spans.append(span)
                span_ws.append((wn, wm))
                beta[A][: N - wn + 1, : M - wm + 1, wn, wm, :] = spans[-1]
                beta[B][wn:, wm:, N - wn, M - wm, :] = spans[-1]

        final = beta[A][0, 0, :, :, NTs]
        top = torch.stack(
            [final[:, i, ln, lm] for i, (ln, lm) in enumerate(lengths)],
            dim=1,
        )
        log_Z = semiring.dot(top, roots)
        return log_Z, (term_use, rules, roots, spans, span_ws)

    def marginals(self, scores, lengths=None, _autograd=True, _raw=False):
        """
        Compute the marginals of a SCFG using bitext CKY.

        Parameters:
            scores : terms : b x n x m x 2 x 2 x T
                     rules : b x NT x (NT+T) x (NT+T)
                     root:   b x NT
            lengths : lengths in batch

        Returns:
            v: b tensor of total sum
            spans: bxnxmx2x2xT terms, (bxNTx(NT+S)x(NT+S)) rules, bxNT roots

        """
        terms, rules, roots = scores
        batch, N_, M_, _, _, T = terms.shape
        _, NT, _, _ = rules.shape
        S = T + NT
        N = N_ - 2
        M = M_ - 2

        v, (term_use, rule_use, root_use, spans, span_ws) = self.logpartition(
            scores, lengths=lengths, force_grad=True
        )

        def marginal(obj, inputs):
            obj = self.semiring.unconvert(obj).sum(dim=0)
            marg = torch.autograd.grad(
                obj,
                inputs,
                create_graph=True,
                only_inputs=True,
                allow_unused=True,
            )

            spans_marg = torch.zeros(
                batch,
                N + 1,
                M + 1,
                N + 1,
                M + 1,
                NT,
                dtype=scores[1].dtype,
                device=scores[1].device,
            )
            span_ls = marg[3:]
            for i, (wn, wm) in enumerate(span_ws):
                x = span_ls[i].sum(dim=0, keepdim=True)
                spans_marg[
                    :, wn, wm, : N - wn + 1, : M - wm + 1
                ] = self.semiring.unconvert(x)

            rule_marg = self.semiring.unconvert(marg[0]).squeeze(1)
            root_marg = self.semiring.unconvert(marg[1])
            term_marg = self.semiring.unconvert(marg[2])

            assert term_marg.shape == (batch, N_, M_, 2, 2, T)
            assert root_marg.shape == (batch, NT)
            assert rule_marg.shape == (batch, NT, S, S)
            return (term_marg, rule_marg, root_marg, spans_marg)

        inputs = (rule_use, root_use, term_use) + tuple(spans)
        if _raw:
            paths = []
            for k in range(v.shape[0]):
                obj = v[k : k + 1]
                marg = marginal(obj, inputs)
                paths.append(marg[-1])
            paths = torch.stack(paths, 0)
            obj = v.sum(dim=0, keepdim=True)
            term_marg, rule_marg, root_marg, _ = marginal(obj, inputs)
            return term_marg, rule_marg, root_marg, paths
        else:
            return marginal(v, inputs)


class SynchCFG(StructDistribution):
    struct = SCKY

    def __init__(self, log_potentials, lengths=None, validate_args=False):
        batch_shape = log_potentials[0].shape[:1]
        event_shape = log_potentials[0].shape[1:]
        self.log_potentials = log_potentials
        self.lengths = lengths
        super(StructDistribution, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )
