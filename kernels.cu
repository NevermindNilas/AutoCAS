// NVRTC source. No toolkit headers: half conversion uses native PTX.
__device__ __forceinline__ float from_half(unsigned short h) {
    float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f;
}
__device__ __forceinline__ unsigned short to_half(float f) {
    unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h;
}
#if HALF
typedef unsigned short pixel;
__device__ __forceinline__ float read(pixel x) { return from_half(x); }
__device__ __forceinline__ pixel write(float x) { return to_half(x); }
__device__ __forceinline__ float rnd(float x) { return from_half(to_half(x)); }
#else
typedef float pixel;
__device__ __forceinline__ float read(pixel x) { return x; }
__device__ __forceinline__ pixel write(float x) { return x; }
__device__ __forceinline__ float rnd(float x) { return x; }
#endif
__device__ __forceinline__ float sat(float x) { return fminf(fmaxf(x, 0.f), 1.f); }
extern "C" __global__ void cas_kernel(const pixel* x, pixel* out, int n, int H, int W,
                                      int C, const pixel* amounts, int mode, float scalar, int bypass) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (unsigned int)n) return;
    int t = (int)index;
    int plane = H * W, p = t % plane, row = p / W, col = p % W;
    int ai = mode == 1 ? 0 : (mode == 2 ? t / (C * plane) : (mode == 3 ? (t / (C * plane))*plane+p : p));
    float amount = mode ? read(amounts[ai]) : rnd(scalar);
    if (bypass && amount < 0.f) { out[t] = x[t]; return; }
    int up = row ? -W : 0, down = row + 1 < H ? W : 0;
    int left = col ? -1 : 0, right = col + 1 < W ? 1 : 0;
    float b = read(x[t+up]), d = read(x[t+left]), e = read(x[t]);
    float f = read(x[t+right]), h = read(x[t+down]);
    float mn = fminf(fminf(fminf(fminf(b,d),e),f),h);
    float mx = fmaxf(fmaxf(fmaxf(fmaxf(b,d),e),f),h);
#if DIAGONALS
    float a = read(x[t+up+left]), c = read(x[t+up+right]);
    float g = read(x[t+down+left]), i = read(x[t+down+right]);
    mn = rnd(mn + fminf(mn, fminf(fminf(fminf(a,c),g),i)));
    mx = rnd(mx + fmaxf(mx, fmaxf(fmaxf(fmaxf(a,c),g),i)));
    float lim = rnd(2.f - mx);
#else
    float lim = rnd(1.f - mx);
#endif
    float amp = rnd(sqrtf(sat(rnd(fminf(mn, lim) / fmaxf(mx, rnd(1e-4f))))));
    float peak = rnd(-rnd(1.f / rnd(8.f - rnd(3.f * sat(amount)))));
    float w = rnd(amp * peak);
    float sum = rnd(rnd(rnd(b+d)+f)+h);
    float den = rnd(rnd(4.f*w)+1.f);
    // addcmul uses opmath accumulation, including for half inputs.
    float value = rnd(fmaf(sum, w, e));
    out[t] = write(sat(rnd(value/den)));
}

__device__ __forceinline__ float luma(const pixel* x, int b, int C, int H, int W, int y, int z) {
    y = max(0,min(H-1,y)); z = max(0,min(W-1,z));
    int p = (b*C)*H*W + y*W + z;
    if (C >= 3) return read(x[p])*.299f + read(x[p+H*W])*.587f + read(x[p+2*H*W])*.114f;
    float v = 0.f;
    for (int c=0;c<C;c++) v += read(x[p+c*H*W]);
    return v/C;
}
extern "C" __global__ void features_kernel(const pixel* x, float* out, int C, int H, int W) {
    __shared__ float sums[16][256];
    float acc[16] = {0};
    int b=blockIdx.x, tid=threadIdx.x;
    for (int s=tid; s<4096; s+=256) {
        int y=(int)(((long long)((s/64)*2+1)*H)/128), z=(int)(((long long)((s%64)*2+1)*W)/128);
        float e=luma(x,b,C,H,W,y,z), u=luma(x,b,C,H,W,y-1,z);
        float d=luma(x,b,C,H,W,y+1,z), l=luma(x,b,C,H,W,y,z-1), r=luma(x,b,C,H,W,y,z+1);
        float a=luma(x,b,C,H,W,y-1,z-1), c=luma(x,b,C,H,W,y-1,z+1);
        float g=luma(x,b,C,H,W,y+1,z-1), i=luma(x,b,C,H,W,y+1,z+1);
        float hi=fmaxf(fmaxf(fmaxf(fmaxf(e,u),d),l),r), lo=fminf(fminf(fminf(fminf(e,u),d),l),r);
        float range=hi-lo, lap=fabsf(u+d+l+r-4.f*e);
        float dx=fabsf(r-l)*.5f, dy=fabsf(d-u)*.5f;
        float chroma=0.f;
        if(C>=3){int p=b*C*H*W+y*W+z; chroma=fabsf(read(x[p])-read(x[p+H*W]))+fabsf(read(x[p+H*W])-read(x[p+2*H*W]));}
        float vals[16]={e,e*e,dx,dy,lap,lap*lap,range,lap/(range+.01f),range*range,
                       (fabsf(i-a)+fabsf(g-c))*.25f,float(e<.02f),float(e>.98f),float(range>.12f),
                       range<.04f?lap:0.f,chroma,fabsf(e-(a+u+c+l+e+r+g+d+i)/9.f)};
        for(int k=0;k<16;k++) acc[k]+=vals[k];
    }
    for(int k=0;k<16;k++) sums[k][tid]=acc[k];
    __syncthreads();
    for(int step=128;step;step>>=1){
        if(tid<step) for(int k=0;k<16;k++) sums[k][tid]+=sums[k][tid+step];
        __syncthreads();
    }
    if(tid==0) for(int k=0;k<16;k++) out[b*16+k]=sums[k][0]/4096.f;
}
// Packed [mean16, inverse_std16, w1(32,16), b1(32), w2(classes,32), b2(classes), amounts(classes)].
extern "C" __global__ void classify_kernel(const float* f, const float* p, pixel* out, int B, int classes) {
    int b=blockIdx.x*blockDim.x+threadIdx.x;
    if(b>=B)return;
    float v[16], hidden[32];
    for(int j=0;j<16;j++)v[j]=(f[b*16+j]-p[j])*p[16+j];
    for(int j=0;j<32;j++){
        float h=p[32+512+j];
        for(int k=0;k<16;k++) h+=p[32+j*16+k]*v[k];
        hidden[j]=fmaxf(0.f,h);
    }
    int offset=32+512+32, best=0;
    float score=-1e30f;
    for(int j=0;j<classes;j++){
        float s=p[offset+classes*32+j];
        for(int k=0;k<32;k++) s+=p[offset+j*32+k]*hidden[k];
        if(s>score){score=s;best=j;}
    }
    out[b]=write(p[offset+classes*32+classes+best]);
}

// Adaptive average pooling: preserve overlapping floor/ceil windows and epsilon
// after averaging. Both numerator and denominator share one coalesced pass.
extern "C" __global__ void tiled_pool_kernel(const float* hf, const float* contrast, float* out, int H, int W, int T) {
    __shared__ float a[256], b[256];
    int tile=blockIdx.x, image=tile/(T*T), ty=(tile/T)%T, tx=tile%T, tid=threadIdx.x;
    int y0=(long long)ty*H/T, y1=((long long)(ty+1)*H+T-1)/T;
    int x0=(long long)tx*W/T, x1=((long long)(tx+1)*W+T-1)/T;
    int width=x1-x0, n=(y1-y0)*width;
    float num=0.f, den=0.f;
    for(int j=tid;j<n;j+=256){
        int index=image*H*W+(y0+j/width)*W+x0+j%width;
        float c=contrast[index]; num+=hf[index]*c; den+=c;
    }
    a[tid]=num;b[tid]=den; __syncthreads();
    for(int step=128;step;step>>=1){if(tid<step){a[tid]+=a[tid+step];b[tid]+=b[tid+step];} __syncthreads();}
    if(tid==0)out[tile]=(a[0]/n)/(b[0]/n+1e-6f);
}
