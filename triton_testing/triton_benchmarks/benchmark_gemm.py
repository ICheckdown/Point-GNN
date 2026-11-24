                          (8192,128,128),(2048,256,128),(65536,64,64)]]


def configurations(max_configs):
    cfg = [
        (16,32,32,4),(16,64,32,4),(32,32,32,4),(32,64,32,4),
        (32,64,64,4),(32,128,32,4),(32,128,64,4),(64,32,32,4),
        (64,64,32,4),(64,64,64,4),(64,128,32,4),(64,128,64,4),
        (128,32,32,4),(128,64,32,4),(128,64,64,4),(128,128,32,4),
        (128,128,64,4),(32,64,32,8),(64,64,32,8),
    ]
    return cfg[:max_configs]


def timing(fn, warmup, repeats, trials):
    """GPU event timing, measures GPU execution rather than CPU enqueue time."""
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    trial_ms=[]
    for _ in range(trials):
        start=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats): fn()
        end.record()
        end.synchronize()
        trial_ms.append(start.elapsed_time(end)/repeats)
    return statistics.median(trial_ms)


def main():
    args=parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is not available. Run on Jetson Orin / NVIDIA CUDA GPU.')
    if args.warmup < 0 or args.repeats < 1 or args.trials < 1:
        raise ValueError('warmup >=0 and repeats/trials >=1 required')
    torch.manual_seed(args.seed)
    device='cuda'
    dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}[args.dtype]
    torch.backends.cuda.matmul.allow_tf32 = args.dtype=='fp32' and args.precision=='tf32'
    # IEEE on torch is higher precision; tf32x3 Triton is approx IEEE, baseline uses IEEE.
    torch.backends.cudnn.benchmark=False
    props=torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name} | torch {torch.__version__} | Triton {triton.__version__}')
    print(f'dtype={args.dtype}, triton_precision={args.precision}, torch_allow_tf32={torch.backends.cuda.matmul.allow_tf32}')
    print('Timing: GPU events; torch baseline = addmm+bias then ReLU; Triton = fused GEMM+bias+ReLU.\n')
    rows=[]
    for sh in load_shapes(args):
        name,m,k,n,w=sh['name'],sh['M'],sh['K'],sh['N'],sh['weight']
        if min(m,k,n)<1: raise ValueError('M, K, N must be positive')
        required=(m*k+k*n+m*n+n)*torch.empty((),dtype=dtype).element_size()/(1024**2)
        if required>args.max_memory_mb:
            print(f'SKIP {name}: estimated tensor storage {required:.1f} MB > limit')
            continue
        print(f'[{name}] M={m}, K={k}, N={n}, inputs+output≈{required:.1f} MiB',flush=True)
        x=torch.randn((m,k),device=device,dtype=dtype)*0.2
        wt=torch.randn((k,n),device=device,dtype=dtype)*0.2
        bias=torch.randn((n,),device=device,dtype=dtype)*0.1
        output=torch.empty((m,n),device=device,dtype=dtype)
        def pytorch_baseline():
            # Use out-of-place relu to avoid changing the object reused by reference validation.
            return torch.relu(torch.addmm(bias,x,wt))
        with torch.no_grad():
            ref=pytorch_baseline()
        base_ms=timing(pytorch_baseline,args.warmup,args.repeats,args.trials)
        best=None
        trials=[]
        for bm,bn,bk,warps in configurations(args.max_configs):
            def run(bm=bm,bn=bn,bk=bk,warps=warps):
                _fused_gemm_relu[(triton.cdiv(m,bm),triton.cdiv(n,bn))](
                    x,wt,bias,output,m,n,k,
                    *x.stride(),*wt.stride(),*output.stride(),
                    bm,bn,bk,args.precision,num_warps=warps)
            try:
                run();torch.cuda.synchronize()
                abs_diff=(output.float()-ref.float()).abs()
                max_err=abs_diff.max().item()
                # Tolerances are sanity checks, not KITTI accuracy validation.
                atol,rtol = ((0.035,0.035) if args.dtype=='fp32' and args.precision=='tf32'
                             else (0.012,0.012) if args.dtype!='fp32' else (0.001,0.001))
                valid=torch.allclose(output.float(),ref.float(),atol=atol,rtol=rtol)
                if not valid:
                    print(f'  WARN config {bm}x{bn}x{bk}/{warps} differs from torch '
                          f'(max abs diff {max_err:.6g}); excluded')
                    continue
                ms=timing(run,args.warmup,args.repeats,args.trials)
                trials.append((ms,bm,bn,bk,warps,max_err))
                if best is None or ms<best[0]: best=(ms,bm,bn,bk,warps,max_err)
            except (RuntimeError, triton.OutOfResources) as exc:
                print(f'  SKIP config {bm}x{bn}x{bk}/{warps}: {str(exc)[:110]}')
        if best is None:
            print(f'  No validated Triton kernel for {name}')
            continue
        ms,bm,bn,bk,warps,err=best
        speedup=base_ms/ms
        print(f'  torch={base_ms:.3f} ms; triton={ms:.3f} ms; speedup={speedup:.3f}x; '
              f'config={bm}x{bn}x{bk}, warps={warps}; max_error={err:.4g}; '
              f'{"PASS" if speedup>=args.threshold else "BELOW THRESHOLD"}',flush=True)
        rows.append(dict(name=name,M=m,K=k,N=n,weight=w,dtype=args.dtype,precision=args.precision,
                         torch_ms=round(base_ms,5),triton_ms=round(ms,5),
                         speedup=round(speedup,4),BM=bm,BN=bn,BK=bk,warps=warps,
                         max_abs_error=round(err,7),exceeds_threshold=speedup>=args.threshold))
        del x,wt,bias,output,ref
    if not rows:
        raise SystemExit('No successful benchmarks.')
    total_torch=sum(r['weight']*r['torch_ms'] for r in rows)
    total_triton=sum(r['weight']*r['triton_ms'] for r in rows)
    weighted_speedup=total_torch/total_triton
    print('\n===== GEMM WEIGHTED SUMMARY =====')
    print(f'Weighted baseline: {total_torch:.3f} ms | Triton: {total_triton:.3f} ms '
          f'| {weighted_speedup:.3f}x | {"MEETS" if weighted_speedup>=args.threshold else "DOES NOT MEET"} '
          f'{args.threshold:.2f}x threshold')
    print('NOTE: This predicts GEMM-only speedup, NOT full Point-GNN end-to-end speedup.')
    print('NOTE: Weights represent frequency of GEMM shape in a forward pass. Set shape-file weights accordingly.')
    with open(args.csv,'w',newline='') as f:
        out=csv.DictWriter(f,fieldnames=list(rows[0].keys()));out.writeheader();out.writerows(rows)
    payload=dict(device=props.name,torch=torch.__version__,triton=triton.__version__,
                 dtype=args.dtype,precision=args.precision,threshold=args.threshold,
                 weighted_torch_ms=total_torch,weighted_triton_ms=total_triton,
                 weighted_gemm_speedup=weighted_speedup,results=rows)
    with open(args.json,'w') as f: json.dump(payload,f,indent=2)
    print(f'Saved {args.csv} and {args.json}')

if __name__=='__main__':
    main()
