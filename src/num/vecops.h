
#ifndef _VECOPS_H
#define _VECOPS_H

extern const struct vec_ops cpu_ops;

struct vec_ops {

	void (*float2double)(bart_dim_t N, double* dst, const float* src);
	void (*double2float)(bart_dim_t N, float* dst, const double* src);
	double (*dot)(bart_dim_t N, const float* vec1, const float* vec2);
	double (*asum)(bart_dim_t N, const float* vec);
	void (*zsum)(bart_dim_t N, _Complex float* vec);
	double (*zl1norm)(bart_dim_t N, const _Complex float* vec);

	_Complex double (*zdot)(bart_dim_t N, const _Complex float* vec1, const _Complex float* vec2);

	void (*axpy)(bart_dim_t N, float* a, float alpha, const float* x);
	void (*axpbz)(bart_dim_t N, float* out, const float a, const float* x, const float b, const float* z);

	void (*pow)(bart_dim_t N, float* dst, const float* src1, const float* src2);
	void (*sqrt)(bart_dim_t N, float* dst, const float* src);
	void (*round)(bart_dim_t N, float* dst, const float* src);

	void (*zle)(bart_dim_t N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
	void (*le)(bart_dim_t N, float* dst, const float* src1, const float* src2);

	void (*add)(bart_dim_t N, float* dst, const float* src1, const float* src2);
	void (*sub)(bart_dim_t N, float* dst, const float* src1, const float* src2);
	void (*mul)(bart_dim_t N, float* dst, const float* src1, const float* src2);
	void (*div)(bart_dim_t N, float* dst, const float* src1, const float* src2);
	void (*fmac)(bart_dim_t N, float* dst, const float* src1, const float* src2);
	void (*fmacD)(bart_dim_t N, double* dst, const float* src1, const float* src2);
	void (*smul)(bart_dim_t N, float alpha, float* dst, const float* src1);
	void (*sadd)(bart_dim_t N, float alpha, float* dst, const float* src1);

	void (*zmul)(bart_dim_t N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
	void (*zdiv)(bart_dim_t N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
	void (*zfmac)(bart_dim_t N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
	void (*zfmacD)(bart_dim_t N, _Complex double* dst, const _Complex float* src1, const _Complex float* src2);
	void (*zmulc)(bart_dim_t N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
	void (*zfmacc)(bart_dim_t N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
	void (*zfmaccD)(bart_dim_t N, _Complex double* dst, const _Complex float* src1, const _Complex float* src2);
	void (*zfsq2)(bart_dim_t N, _Complex float* dst, const _Complex float* src1);

	void (*zsmul)(bart_dim_t N, _Complex float val, _Complex float* dst, const _Complex float* src1);
	void (*zsadd)(bart_dim_t N, _Complex float val, _Complex float* dst, const _Complex float* src1);

	void (*zpow)(bart_dim_t N,  _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
	void (*zphsr)(bart_dim_t N, _Complex float* dst, const _Complex float* src);
	void (*zconj)(bart_dim_t N, _Complex float* dst, const _Complex float* src);
	void (*zexpj)(bart_dim_t N, _Complex float* dst, const _Complex float* src);
	void (*zexp)(bart_dim_t N, _Complex float* dst, const _Complex float* src);
	void (*zlog)(bart_dim_t N, _Complex float* dst, const _Complex float* src);
	void (*zarg)(bart_dim_t N, _Complex float* dst, const _Complex float* src);
	void (*zabs)(bart_dim_t N, _Complex float* dst, const _Complex float* src);
	void (*zatanr)(bart_dim_t N, _Complex float* dst, const _Complex float* src);
	void (*zatan2r)(bart_dim_t N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);

	void (*exp)(bart_dim_t N, float* dst, const float* src);
	void (*log)(bart_dim_t N, float* dst, const float* src);

	void (*zsin)(bart_dim_t N, _Complex float* dst, const _Complex float* src);
	void (*zcos)(bart_dim_t N, _Complex float* dst, const _Complex float* src);
	void (*zasin)(bart_dim_t N, _Complex float* dst, const _Complex float* src);
	void (*zacos)(bart_dim_t N, _Complex float* dst, const _Complex float* src);
	void (*zacosr)(bart_dim_t N, _Complex float* dst, const _Complex float* src);

	void (*zsinh)(bart_dim_t N, _Complex float* dst, const _Complex float* src);
	void (*zcosh)(bart_dim_t N, _Complex float* dst, const _Complex float* src);

	void (*zcmp)(bart_dim_t N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
	void (*zdiv_reg)(bart_dim_t N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2, _Complex float lambda);
	void (*zfftmod)(bart_dim_t N, _Complex float* dst, const _Complex float* src, int n, bool inv, double phase);

	void (*zmax)(bart_dim_t N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
	void (*zsmax)(bart_dim_t N, float alpha, _Complex float* dst, const _Complex float* src);
	void (*zsmin)(bart_dim_t N, float alpha, _Complex float* dst, const _Complex float* src);

	void (*smax)(bart_dim_t N, float val, float* dst, const float* src1);
	void (*max)(bart_dim_t N, float* dst, const float* src1, const float* src2);
	void (*min)(bart_dim_t N, float* dst, const float* src1, const float* src2);

	void (*zsoftthresh_half)(bart_dim_t N, float lambda,  _Complex float* dst, const _Complex float* src);
	void (*zsoftthresh)(bart_dim_t N, float lambda,  _Complex float* dst, const _Complex float* src);
	void (*softthresh_half)(bart_dim_t N, float lambda,  float* dst, const float* src);
	void (*softthresh)(bart_dim_t N, float lambda,  float* dst, const float* src);
//	void (*swap)(long N, float* a, float* b);
	void (*zhardthresh)(bart_dim_t N, int k, _Complex float* d, const _Complex float* x);
	void (*zhardthresh_mask)(bart_dim_t N, int k, _Complex float* d, const _Complex float* x);

	void (*pdf_gauss)(bart_dim_t N, float mu, float sig, float* dst, const float* src);

	void (*real)(bart_dim_t N, float* dst, const _Complex float* src);
	void (*imag)(bart_dim_t N, float* dst, const _Complex float* src);
	void (*zcmpl_real)(bart_dim_t N, _Complex float* dst, const float* src);
	void (*zcmpl_imag)(bart_dim_t N, _Complex float* dst, const float* src);
	void (*zcmpl)(bart_dim_t N, _Complex float* dst, const float* real_src, const float* imag_src);

	void (*zfill)(bart_dim_t N, _Complex float val, _Complex float* dst);

	void (*zsetnanzero)(bart_dim_t N, _Complex float* dst, const _Complex float* src);
};

#endif

