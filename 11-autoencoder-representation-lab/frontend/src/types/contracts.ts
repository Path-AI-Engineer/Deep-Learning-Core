export type Metrics = {mse:number; mae:number; psnr:number|null; ssim:number};
export type Sample = {sample_id:string; label:number; class_name:string; image:string; split:string};
export type ModelRow = {
  model_id:string;
  version:string;
  active:boolean;
  validation_mse:number;
  parameters:number;
  reconstruction:Metrics;
  representation:{linear_probe:{accuracy:number;macro_f1:number}|null};
  robustness:Record<string, unknown>;
  capabilities:{reconstruct:boolean;denoise:boolean;encode:boolean;decode_coordinates:boolean};
};
export type Reconstruction = {
  sample_id:string; label:number; class_name:string; model_id:string; model_version:string;
  original:string; reconstruction:string; absolute_error:string; latent:number[];
  metrics:Metrics; warning:string;
};
export type LatentPoint = {
  sample_id:string; label:number; class_name:string; x:number; y:number;
  image:string; reconstruction:string; neighbors:{sample_id:string;distance:number}[];
};
