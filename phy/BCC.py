import numpy as np

class ViterbiCodec:
    def __init__(self, constraint, polynomials, puncpat="1"):
        self.constraint = constraint
        self.polynomials = polynomials
        self.puncpat = puncpat
        self.outputs = []
        self.initialize_outputs()

    def num_parity_bits(self):
        return len(self.polynomials)

    def next_state(self, current_state, input_bit):
        return (current_state >> 1) | (input_bit << (self.constraint - 2))

    def output(self, current_state, input_bit):
        return self.outputs[current_state | (input_bit << (self.constraint - 1))]

    def encode(self, bits):
        encoded = ""
        state = 0

        # Encode the message bits
        for c in bits:
            assert c in '01'
            input_bit = int(c)
            encoded += self.output(state, input_bit)
            state = self.next_state(state, input_bit)

        if self.puncpat:
            return self.puncturing(encoded)

        return encoded

    def initialize_outputs(self):
        self.outputs = [""] * (1 << self.constraint)
        for i in range(len(self.outputs)):
            for polynomial in self.polynomials:
                output = 0
                polynomial = self.reverse_bits(self.constraint, polynomial)
                input_bits = i
                for _ in range(self.constraint):
                    output ^= (input_bits & 1) & (polynomial & 1)
                    polynomial >>= 1
                    input_bits >>= 1
                self.outputs[i] += "1" if output else "0"

    def hamming_distance(self, x, y):
        assert len(x) == len(y)
        distance = 0
        for i in range (len(x)):
            if (x[i] == '-' or y[i] == '-'):
                continue
            distance += x[i]!=y[i]
        return distance

    def branch_metric(self, bits, source_state, target_state):
        assert len(bits) == self.num_parity_bits()
        output = self.output(source_state, target_state >> (self.constraint - 2))
        return self.hamming_distance(bits, output)

    def path_metric(self, bits, prev_path_metrics, state):
        s = (state & ((1 << (self.constraint - 2)) - 1)) << 1
        source_state1, source_state2 = s | 0, s | 1

        pm1 = prev_path_metrics[source_state1]
        if pm1 < float('inf'):
            pm1 += self.branch_metric(bits, source_state1, state)

        pm2 = prev_path_metrics[source_state2]
        if pm2 < float('inf'):
            pm2 += self.branch_metric(bits, source_state2, state)

        if pm1 <= pm2:
            return pm1, source_state1
        else:
            return pm2, source_state2

    def update_path_metrics(self, bits, path_metrics, trellis):
        new_path_metrics = [float('inf')] * len(path_metrics)
        new_trellis_column = [0] * len(path_metrics)
        for i in range(len(path_metrics)):
            pm, source_state = self.path_metric(bits, path_metrics, i)
            new_path_metrics[i] = pm
            new_trellis_column[i] = source_state
        path_metrics[:] = new_path_metrics
        trellis.append(new_trellis_column)
        return path_metrics,trellis

    def decode(self, bits):
        depunctured = self.depuncturing(bits) + "0" * self.num_parity_bits() * (self.constraint - 1)
        trellis = []
        path_metrics = [float('inf')] * (1 << (self.constraint - 1))
        path_metrics[0] = 0

        for i in range(0, len(depunctured), self.num_parity_bits()):
            current_bits = depunctured[i:i + self.num_parity_bits()]
            if (len(current_bits)<self.num_parity_bits()):
                current_bits.ljust(self.num_parity_bits()-len(current_bits), '0')
            path_metrics,trellis = self.update_path_metrics(current_bits, path_metrics, trellis)

        decoded = ""
        state = int(np.argmin(path_metrics))
        for i in range(len(trellis)-1,-1,-1):
            decoded+= '1' if state>>(self.constraint -2) else '0'
            state = trellis[i][state]
        decoded = decoded[::-1]

        return decoded[:len(decoded)-self.constraint+1]

    def puncturing(self, bits):
        punctured = ""
        index = 0
        for bit in bits:
            if self.puncpat[index] == '1':
                punctured += bit
            index = (index + 1) % len(self.puncpat)
        return punctured

    def depuncturing(self, bits):
        depunctured = ""
        index = 0
        i=0
        while index<len(bits):
            for j in range (len(self.puncpat)):
                if self.puncpat[j]=='1':
                    if index>=len(bits):
                        depunctured+='0'
                    else:
                        depunctured+=bits[index]
                        index+=1
                else:
                    depunctured+='-'
            i+=len(self.puncpat)
        return depunctured

    @staticmethod
    def reverse_bits(num_bits: int, input_bits: int):
        output = 0
        for _ in range(num_bits):
            output = (output << 1) | (input_bits & 1)
            input_bits >>= 1
        return output
