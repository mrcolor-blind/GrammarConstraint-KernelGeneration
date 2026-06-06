const BASE_URL = 'http://localhost:8000/api/v1';

export interface TranslatePayload {
  source_code: string;
  provider?: string;
  model?: string;
  dims?: Record<string, number>;
  gpu_validate?: boolean;
}

export interface TranslateResponse {
  job_id: string;
  status: string;
  provider: string;
  model: string;
  run_id?: string;
  source_code?: string;
  generated_code: string | null;
  validation: {
    passed: boolean;
    errors: string[];
    warnings: string[];
  };
  gpu_validation: any;
  errors: string[];
}

export interface GpuValidateResponse {
  compilation_pass: boolean;
  execution_pass: boolean;
  output_shape?: string;
  device?: string;
  errors: string[];
}

export interface CompareResponse {
  compilation_pass: boolean;
  accuracy_pass: boolean;
  max_diff?: number;
  speedup?: number;
  ref_time_ms?: number;
  gen_time_ms?: number;
  suggest_replacement: boolean;
  reason: string;
  errors: string[];
  device?: string;
}

export interface EvaluateResponse {
  job_id: string;
  accuracy_pass: boolean;
  max_error: number;
  speedup: number;
  errors: string[];
}

export interface JobSummary {
  job_id: string;
  status: string;
  type: string;
  provider: string;
  model: string;
  run_id?: string;
  function_name?: string;
  created_at?: string;
}

export interface JobListResponse {
  total: number;
  limit: number;
  offset: number;
  items: JobSummary[];
}

export interface JobDetail extends JobSummary {
  source_code?: string;
  generated_code?: string | null;
  validation?: any;
  gpu_validation?: any;
  errors?: string[];
}

export class TritonClient {
  private async _post<T>(endpoint: string, body: object, timeoutMs: number): Promise<T> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(`${BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      clearTimeout(timeout);

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`HTTP ${response.status}: ${text}`);
      }

      return (await response.json()) as T;
    } catch (err: any) {
      clearTimeout(timeout);
      if (err.name === 'AbortError') {
        throw new Error('La petición ha excedido el tiempo de espera.');
      }
      throw err;
    }
  }

  async translate(payload: TranslatePayload): Promise<TranslateResponse> {
    return this._post<TranslateResponse>('/translate', payload, 120000); // 120s
  }

  async gpuValidate(jobId: string): Promise<GpuValidateResponse> {
    return this._post<GpuValidateResponse>(`/jobs/${jobId}/gpu-validate`, {}, 360000); // 360s
  }

  async compare(jobId: string): Promise<CompareResponse> {
    return this._post<CompareResponse>(`/jobs/${jobId}/compare`, {}, 600000); // 600s
  }

  async evaluate(jobId: string, dims: Record<string, number>): Promise<EvaluateResponse> {
    return this._post<EvaluateResponse>('/evaluate', { job_id: jobId, dims }, 120000); // 120s
  }

  async listRuns(limit: number = 20): Promise<JobListResponse> {
    const response = await fetch(`${BASE_URL}/runs?limit=${limit}`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`HTTP ${response.status}: ${text}`);
    }
    return (await response.json()) as JobListResponse;
  }

  async getRun(jobId: string): Promise<JobDetail> {
    const response = await fetch(`${BASE_URL}/runs/${jobId}`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`HTTP ${response.status}: ${text}`);
    }
    return (await response.json()) as JobDetail;
  }
}

export default new TritonClient();
