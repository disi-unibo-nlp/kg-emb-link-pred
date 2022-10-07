from pykg2vec.models.KGMeta import PointwiseModel

class DualE(PointwiseModel):
    """
        `Dual Quaternion Knowledge Graph Embeddings`_

        Args:
            config (object): Model configuration parameters.

        .. _code: https://github.com/Lion-ZS/DualE

        .. _Dual Quaternion Knowledge Graph Embeddings:
            https://ojs.aaai.org/index.php/AAAI/article/view/16850

    """

    def __init__(self, **kwargs):
        super(DualE, self).__init__(self.__class__.__name__.lower())
        param_list = ["tot_entity", "tot_relation", "hidden_size", "lmbda"]
        param_dict = self.load_params(param_list, kwargs)
        self.__dict__.update(param_dict)

        num_total_ent = self.tot_entity
        num_total_rel = self.tot_relation
        k = self.hidden_size

        self.ent_1_embedding = NamedEmbedding("ent_s_embedding", num_total_ent, k)
        self.ent_2_embedding = NamedEmbedding("ent_x_embedding", num_total_ent, k)
        self.ent_3_embedding = NamedEmbedding("ent_y_embedding", num_total_ent, k)
        self.ent_4_embedding = NamedEmbedding("ent_z_embedding", num_total_ent, k)
        self.ent_5_embedding = NamedEmbedding("ent_s_embedding", num_total_ent, k)
        self.ent_6_embedding = NamedEmbedding("ent_x_embedding", num_total_ent, k)
        self.ent_7_embedding = NamedEmbedding("ent_y_embedding", num_total_ent, k)
        self.ent_8_embedding = NamedEmbedding("ent_z_embedding", num_total_ent, k)
        self.rel_1_embedding = NamedEmbedding("rel_s_embedding", num_total_rel, k)
        self.rel_2_embedding = NamedEmbedding("rel_x_embedding", num_total_rel, k)
        self.rel_3_embedding = NamedEmbedding("rel_y_embedding", num_total_rel, k)
        self.rel_4_embedding = NamedEmbedding("rel_z_embedding", num_total_rel, k)
        self.rel_5_embedding = NamedEmbedding("rel_s_embedding", num_total_rel, k)
        self.rel_6_embedding = NamedEmbedding("rel_x_embedding", num_total_rel, k)
        self.rel_7_embedding = NamedEmbedding("rel_y_embedding", num_total_rel, k)
        self.rel_8_embedding = NamedEmbedding("rel_z_embedding", num_total_rel, k)
        self.rel_w_embedding = NamedEmbedding("rel_w_embedding", num_total_rel, k)

        self.loss = Criterion.pointwise_logistic
        self.fc = nn.Linear(100, 50, bias=False)
        self.ent_dropout = nn.Dropout(0)
        self.rel_dropout = nn.Dropout(0)
        self.bn = nn.BatchNorm1d(k)

        rel_1, rel_2, rel_3, rel_4, rel_5, rel_6, rel_7, rel_8 = DualE._quaternion_init(self.tot_entity, self.hidden_size)
        rel_1, rel_2, rel_3, rel_4, rel_5, rel_6, rel_7, rel_8 = torch.from_numpy(rel_1), torch.from_numpy(rel_2), torch.from_numpy(rel_3), torch.from_numpy(rel_4), torch.from_numpy(rel_5), torch.from_numpy(rel_6), torch.from_numpy(rel_7), torch.from_numpy(rel_8)       
        self.ent_1_embedding.weight.data = rel_1.type_as(self.ent_1_embedding.weight.data)
        self.ent_2_embedding.weight.data = rel_2.type_as(self.ent_2_embedding.weight.data)
        self.ent_3_embedding.weight.data = rel_3.type_as(self.ent_3_embedding.weight.data)
        self.ent_4_embedding.weight.data = rel_4.type_as(self.ent_4_embedding.weight.data)
        self.ent_5_embedding.weight.data = rel_5.type_as(self.ent_5_embedding.weight.data)
        self.ent_6_embedding.weight.data = rel_6.type_as(self.ent_6_embedding.weight.data)
        self.ent_7_embedding.weight.data = rel_7.type_as(self.ent_7_embedding.weight.data)
        self.ent_8_embedding.weight.data = rel_8.type_as(self.ent_8_embedding.weight.data)

        ent_1, ent_2, ent_3, ent_4, ent_5, ent_6, ent_7, ent_8 = DualE._quaternion_init(self.tot_entity, self.hidden_size)
        ent_1, ent_2, ent_3, ent_4, ent_5, ent_6, ent_7, ent_8 = torch.from_numpy(ent_1), torch.from_numpy(ent_2), torch.from_numpy(ent_3), torch.from_numpy(ent_4), torch.from_numpy(ent_5), torch.from_numpy(ent_6), torch.from_numpy(ent_7), torch.from_numpy(ent_8)       
        self.rel_1_embedding.weight.data = ent_1.type_as(self.rel_1_embedding.weight.data)
        self.rel_2_embedding.weight.data = ent_2.type_as(self.rel_2_embedding.weight.data)
        self.rel_3_embedding.weight.data = ent_3.type_as(self.rel_3_embedding.weight.data)
        self.rel_4_embedding.weight.data = ent_4.type_as(self.rel_4_embedding.weight.data)
        self.rel_5_embedding.weight.data = ent_5.type_as(self.rel_5_embedding.weight.data)
        self.rel_6_embedding.weight.data = ent_6.type_as(self.rel_6_embedding.weight.data)
        self.rel_7_embedding.weight.data = ent_7.type_as(self.rel_7_embedding.weight.data)
        self.rel_8_embedding.weight.data = ent_8.type_as(self.rel_8_embedding.weight.data)

        nn.init.xavier_uniform_(self.ent_1_embedding.weight.data)
        nn.init.xavier_uniform_(self.ent_2_embedding.weight.data)
        nn.init.xavier_uniform_(self.ent_3_embedding.weight.data)
        nn.init.xavier_uniform_(self.ent_4_embedding.weight.data)
        nn.init.xavier_uniform_(self.ent_5_embedding.weight.data)
        nn.init.xavier_uniform_(self.ent_6_embedding.weight.data)
        nn.init.xavier_uniform_(self.ent_7_embedding.weight.data)
        nn.init.xavier_uniform_(self.ent_8_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_1_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_2_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_3_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_4_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_5_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_6_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_7_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_8_embedding.weight.data)        
        nn.init.xavier_uniform_(self.rel_w_embedding.weight.data)

        self.parameter_list = [
            self.ent_1_embedding,
            self.ent_2_embedding,
            self.ent_3_embedding,
            self.ent_4_embedding,
            self.ent_5_embedding,
            self.ent_6_embedding,
            self.ent_7_embedding,
            self.ent_8_embedding,
            self.rel_1_embedding,
            self.rel_2_embedding,
            self.rel_3_embedding,
            self.rel_4_embedding,
            self.rel_5_embedding,
            self.rel_6_embedding,
            self.rel_7_embedding,
            self.rel_8_embedding,
            self.rel_w_embedding,
        ]

        

    def embed(self, h, r, t):
        ent_1_emb_h = self.ent_1_embedding(h)
        ent_2_emb_h = self.ent_2_embedding(h)
        ent_3_emb_h = self.ent_3_embedding(h)
        ent_4_emb_h = self.ent_4_embedding(h)
        ent_5_emb_h = self.ent_5_embedding(h)
        ent_6_emb_h = self.ent_6_embedding(h)
        ent_7_emb_h = self.ent_7_embedding(h)
        ent_8_emb_h = self.ent_8_embedding(h)

        ent_1_emb_t = self.ent_1_embedding(t)
        ent_2_emb_t = self.ent_2_embedding(t)
        ent_3_emb_t = self.ent_3_embedding(t)
        ent_4_emb_t = self.ent_4_embedding(t)
        ent_5_emb_t = self.ent_5_embedding(t)
        ent_6_emb_t = self.ent_6_embedding(t)
        ent_7_emb_t = self.ent_7_embedding(t)
        ent_8_emb_t = self.ent_8_embedding(t)

        rel_1_emb_r = self.rel_1_embedding(r)
        rel_2_emb_r = self.rel_2_embedding(r)
        rel_3_emb_r = self.rel_3_embedding(r)
        rel_4_emb_r = self.rel_4_embedding(r)
        rel_5_emb_r = self.rel_5_embedding(r)
        rel_6_emb_r = self.rel_6_embedding(r)
        rel_7_emb_r = self.rel_7_embedding(r)
        rel_8_emb_r = self.rel_8_embedding(r)

        return ent_1_emb_h, ent_2_emb_h, ent_3_emb_h, ent_4_emb_h, ent_5_emb_h, ent_6_emb_h, ent_7_emb_h, ent_8_emb_h, ent_1_emb_t, ent_2_emb_t, ent_3_emb_t, ent_4_emb_t, ent_5_emb_t, ent_6_emb_t, ent_7_emb_t, ent_8_emb_t, rel_1_emb_r, rel_2_emb_r, rel_3_emb_r, rel_4_emb_r, rel_5_emb_r, rel_6_emb_r, rel_7_emb_r, rel_8_emb_r

    #Calculate the Dual Hamiltonian product
    def _omult(self, a_0, a_1, a_2, a_3, b_0, b_1, b_2, b_3, c_0, c_1, c_2, c_3, d_0, d_1, d_2, d_3):
        h_0 = a_0*c_0 - a_1*c_1 - a_2*c_2 - a_3*c_3
        h1_0 = a_0*d_0 + b_0*c_0 - a_1*d_1 - b_1*c_1 - a_2*d_2 - b_2*c_2 - a_3*d_3 - b_3*c_3
        h_1 = a_0*c_1 + a_1*c_0 + a_2*c_3 - a_3*c_2
        h1_1 = a_0*d_1 + b_0*c_1 + a_1*d_0 + b_1*c_0 + a_2*d_3 + b_2*c_3 - a_3*d_2 - b_3*c_2
        h_2 = a_0*c_2 - a_1*c_3 + a_2*c_0 + a_3*c_1
        h1_2 = a_0*d_2 + b_0*c_2 - a_1*d_3 - b_1*c_3 + a_2*d_0 + b_2*c_0 + a_3*d_1 + b_3*c_1
        h_3 = a_0*c_3 + a_1*c_2 - a_2*c_1 + a_3*c_0
        h1_3 = a_0*d_3 + b_0*c_3 + a_1*d_2 + b_1*c_2 - a_2*d_1 - b_2*c_1 + a_3*d_0 + b_3*c_0
        return  (h_0, h_1, h_2, h_3, h1_0, h1_1, h1_2, h1_3)  
    
    #Normalization of relationship embedding
    def _onorm(self, r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8):
        denominator_0 = r_1 ** 2 + r_2 ** 2 + r_3 ** 2 + r_4 ** 2
        denominator_1 = torch.sqrt(denominator_0)
        #denominator_2 = torch.sqrt(r_5 ** 2 + r_6 ** 2 + r_7 ** 2 + r_8 ** 2)
        deno_cross = r_5 * r_1 + r_6 * r_2 + r_7 * r_3 + r_8 * r_4
        r_5 = r_5 - deno_cross / denominator_0 * r_1
        r_6 = r_6 - deno_cross / denominator_0 * r_2
        r_7 = r_7 - deno_cross / denominator_0 * r_3
        r_8 = r_8 - deno_cross / denominator_0 * r_4
        r_1 = r_1 / denominator_1
        r_2 = r_2 / denominator_1
        r_3 = r_3 / denominator_1
        r_4 = r_4 / denominator_1
        #r_5 = r_5 / denominator_2
        #r_6 = r_6 / denominator_2
        #r_7 = r_7 / denominator_2
        #r_8 = r_8 / denominator_2
        return r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8

    #Calculate the inner product of the head entity and the relationship Hamiltonian product and the tail entity
    def forward(self, h, r, t):
        ent_1_emb_h, ent_2_emb_h, ent_3_emb_h, ent_4_emb_h, ent_5_emb_h, ent_6_emb_h, ent_7_emb_h, ent_8_emb_h, ent_1_emb_t, ent_2_emb_t, ent_3_emb_t, ent_4_emb_t, ent_5_emb_t, ent_6_emb_t, ent_7_emb_t, ent_8_emb_t, rel_1_emb_r, rel_2_emb_r, rel_3_emb_r, rel_4_emb_r, rel_5_emb_r, rel_6_emb_r, rel_7_emb_r, rel_8_emb_r = self.embed(h, r, t)
        rel_1_emb_r, rel_2_emb_r, rel_3_emb_r, rel_4_emb_r, rel_5_emb_r, rel_6_emb_r, rel_7_emb_r, rel_8_emb_r = self._onorm(rel_1_emb_r, rel_2_emb_r, rel_3_emb_r, rel_4_emb_r, rel_5_emb_r, rel_6_emb_r, rel_7_emb_r, rel_8_emb_r)
        a, b, c, d, e, f, g, h = self._omult(ent_1_emb_h, ent_2_emb_h, ent_3_emb_h, ent_4_emb_h, ent_5_emb_h, ent_6_emb_h, ent_7_emb_h, ent_8_emb_h, rel_1_emb_r, rel_2_emb_r, rel_3_emb_r, rel_4_emb_r, rel_5_emb_r, rel_6_emb_r, rel_7_emb_r, rel_8_emb_r)
        score_r = a * ent_1_emb_t + b * ent_2_emb_t + c * ent_3_emb_t + d * ent_4_emb_t + e * ent_5_emb_t + f * ent_6_emb_t + g * ent_7_emb_t + h * ent_8_emb_t

        return -torch.sum(score_r, -1)

    def get_reg(self, h, r, t, reg_type='f2'):
        ent_1_emb_h, ent_2_emb_h, ent_3_emb_h, ent_4_emb_h, ent_5_emb_h, ent_6_emb_h, ent_7_emb_h, ent_8_emb_h, ent_1_emb_t, ent_2_emb_t, ent_3_emb_t, ent_4_emb_t, ent_5_emb_t, ent_6_emb_t, ent_7_emb_t, ent_8_emb_t, rel_1_emb_r, rel_2_emb_r, rel_3_emb_r, rel_4_emb_r, rel_5_emb_r, rel_6_emb_r, rel_7_emb_r, rel_8_emb_r = self.embed(h, r, t)
        if reg_type.lower() == 'f2':
            regul = (torch.mean(torch.abs(ent_1_emb_h) ** 2)
                     + torch.mean(torch.abs(ent_2_emb_h) ** 2)
                     + torch.mean(torch.abs(ent_3_emb_h) ** 2)
                     + torch.mean(torch.abs(ent_4_emb_h) ** 2)
                     + torch.mean(torch.abs(ent_5_emb_h) ** 2)
                     + torch.mean(torch.abs(ent_6_emb_h) ** 2)
                     + torch.mean(torch.abs(ent_7_emb_h) ** 2)
                     + torch.mean(torch.abs(ent_8_emb_h) ** 2)
                     + torch.mean(torch.abs(ent_1_emb_t) ** 2)
                     + torch.mean(torch.abs(ent_2_emb_t) ** 2)
                     + torch.mean(torch.abs(ent_3_emb_t) ** 2)
                     + torch.mean(torch.abs(ent_4_emb_t) ** 2)
                     + torch.mean(torch.abs(ent_5_emb_t) ** 2)
                     + torch.mean(torch.abs(ent_6_emb_t) ** 2)
                     + torch.mean(torch.abs(ent_7_emb_t) ** 2)
                     + torch.mean(torch.abs(ent_8_emb_t) ** 2)
                     )
            regul2 = (torch.mean(torch.abs(rel_1_emb_r) ** 2)
                      + torch.mean(torch.abs(rel_2_emb_r) ** 2)
                      + torch.mean(torch.abs(rel_3_emb_r) ** 2)
                      + torch.mean(torch.abs(rel_4_emb_r) ** 2)
                      + torch.mean(torch.abs(rel_5_emb_r) ** 2)
                      + torch.mean(torch.abs(rel_6_emb_r) ** 2)
                      + torch.mean(torch.abs(rel_7_emb_r) ** 2)
                      + torch.mean(torch.abs(rel_8_emb_r) ** 2)
                      )
        elif reg_type.lower() == 'n3':
            regul = (torch.mean(torch.abs(ent_1_emb_h) ** 3)
                     + torch.mean(torch.abs(ent_2_emb_h) ** 3)
                     + torch.mean(torch.abs(ent_3_emb_h) ** 3)
                     + torch.mean(torch.abs(ent_4_emb_h) ** 3)
                     + torch.mean(torch.abs(ent_5_emb_h) ** 3)
                     + torch.mean(torch.abs(ent_6_emb_h) ** 3)
                     + torch.mean(torch.abs(ent_7_emb_h) ** 3)
                     + torch.mean(torch.abs(ent_8_emb_h) ** 3)
                     + torch.mean(torch.abs(ent_1_emb_t) ** 3)
                     + torch.mean(torch.abs(ent_2_emb_t) ** 3)
                     + torch.mean(torch.abs(ent_3_emb_t) ** 3)
                     + torch.mean(torch.abs(ent_4_emb_t) ** 3)
                     + torch.mean(torch.abs(ent_5_emb_t) ** 3)
                     + torch.mean(torch.abs(ent_6_emb_t) ** 3)
                     + torch.mean(torch.abs(ent_7_emb_t) ** 3)
                     + torch.mean(torch.abs(ent_8_emb_t) ** 3)
                     )
            regul2 = (torch.mean(torch.abs(rel_1_emb_r) ** 3)
                      + torch.mean(torch.abs(rel_2_emb_r) ** 3)
                      + torch.mean(torch.abs(rel_3_emb_r) ** 3)
                      + torch.mean(torch.abs(rel_4_emb_r) ** 3)
                      + torch.mean(torch.abs(rel_5_emb_r) ** 3)
                      + torch.mean(torch.abs(rel_6_emb_r) ** 3)
                      + torch.mean(torch.abs(rel_7_emb_r) ** 3)
                      + torch.mean(torch.abs(rel_8_emb_r) ** 3)
                      )
        else:
            raise NotImplementedError('Unknown regularizer type: %s' % reg_type)

        return self.lmbda * (regul + regul2)

    @staticmethod
    def _quaternion_init(in_features, out_features, criterion='he'):

        fan_in = in_features
        fan_out = out_features

        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
        rng = RandomState(2020)

        kernel_shape = (in_features, out_features)

        number_of_weights = np.prod(kernel_shape)
        v_i = np.random.uniform(0.0, 1.0, number_of_weights)
        v_j = np.random.uniform(0.0, 1.0, number_of_weights)
        v_k = np.random.uniform(0.0, 1.0, number_of_weights)

        for i in range(0, number_of_weights):
            norm = np.sqrt(v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2) + 0.0001
            v_i[i] /= norm
            v_j[i] /= norm
            v_k[i] /= norm
        v_i = v_i.reshape(kernel_shape)
        v_j = v_j.reshape(kernel_shape)
        v_k = v_k.reshape(kernel_shape)

        modulus = rng.uniform(low=-s, high=s, size=kernel_shape)

        # Calculate the three parts about t
        kernel_shape1 = (in_features, out_features)
        number_of_weights1 = np.prod(kernel_shape1)
        t_i = np.random.uniform(0.0, 1.0, number_of_weights1)
        t_j = np.random.uniform(0.0, 1.0, number_of_weights1)
        t_k = np.random.uniform(0.0, 1.0, number_of_weights1)

        # Purely imaginary quaternions unitary
        for i in range(0, number_of_weights1):
            norm1 = np.sqrt(t_i[i] ** 2 + t_j[i] ** 2 + t_k[i] ** 2) + 0.0001
            t_i[i] /= norm1
            t_j[i] /= norm1
            t_k[i] /= norm1
        t_i = t_i.reshape(kernel_shape1)
        t_j = t_j.reshape(kernel_shape1)
        t_k = t_k.reshape(kernel_shape1)
        tmp_t = rng.uniform(low=-s, high=s, size=kernel_shape1)

        phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
        phase1 = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape1)

        weight_r = modulus * np.cos(phase)
        weight_i = modulus * v_i * np.sin(phase)
        weight_j = modulus * v_j * np.sin(phase)
        weight_k = modulus * v_k * np.sin(phase)

        wt_i = tmp_t * t_i * np.sin(phase1)
        wt_j = tmp_t * t_j * np.sin(phase1)
        wt_k = tmp_t * t_k * np.sin(phase1)

        i_0 = weight_r
        i_1 = weight_i
        i_2 = weight_j
        i_3 = weight_k
        i_4 = (-wt_i*weight_i-wt_j*weight_j-wt_k*weight_k)/2
        i_5 = (wt_i*weight_r+wt_j*weight_k-wt_k*weight_j)/2
        i_6 = (-wt_i*weight_k+wt_j*weight_r+wt_k*weight_i)/2
        i_7 = (wt_i*weight_j-wt_j*weight_i+wt_k*weight_r)/2

        return i_0, i_1, i_2, i_3, i_4, i_5, i_6, i_7