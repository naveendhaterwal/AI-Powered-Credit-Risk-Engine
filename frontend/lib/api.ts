export interface BorrowerInput {
  full_name: string;
  age: number;
  monthly_income: number;
  employment_type: "Salaried" | "Self-Employed" | "Business";
  credit_score: number;
  existing_loan_amount?: number;
  existing_emi_monthly?: number;
  loan_amount_requested: number;
  loan_purpose: "Home" | "Auto" | "Personal" | "Business";
  loan_tenure_months?: number;
}

export interface RiskAnalysis {
  risk_level: string;
  ml_risk_score: number;
  top_risk_factors: string[];
  positive_factors: string[];
  financial_sanity_score: number;
  policy_risk_score: number;
  final_ai_score: number;
  arbitration_summary: string;
  prediction_confidence: number;
  disagreement_score: number;
}

export interface PolicyMatch {
  rule_name: string;
  rule_text: string;
  status: "Passed" | "Failed" | "Warning";
  source?: string;
  score?: number;
}

export interface AgentInteraction {
  agent: string;
  prompt: string;
  response: string;
  system_prompt?: string;
  metadata?: any;
  error?: string;
}

export interface WorkflowTraceStep {
  step: string;
  node: string;
  status: "completed" | "failed";
  started_at: string;
  ended_at: string;
  duration_ms: number;
  model: string;
  source: string;
  note?: string;
  input: Record<string, any>;
  output: Record<string, any>;
}

export interface PredictionResponse {
  borrower_name: string;
  request_id: string;
  risk_analysis: RiskAnalysis;
  policy_retrieval: {
    rules_checked: number;
    policies_matched: PolicyMatch[];
  };
  lending_decision: {
    recommendation: "Approve" | "Approve with Lower Amount" | "Manual Review" | "Reject";
    primary_reason: string;
    secondary_reasons: string[];
    suggested_action: string;
    manual_review_needed: boolean;
  };
  foir: number;
  dti: number;
  proposed_emi: number;
  agent_interactions: AgentInteraction[];
  score_breakdown: Record<string, any>;
  workflow_trace: WorkflowTraceStep[];
}

// NEXT_PUBLIC_API_URL = your Render backend URL (set in Vercel env vars)
// Falls back to localhost:8000 for local development
const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

export async function predictRisk(data: BorrowerInput): Promise<PredictionResponse> {
  const response = await fetch(`${API_BASE_URL}/predict/risk`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    let errorMessage = "Prediction failed";
    
    if (errorData.detail) {
      if (typeof errorData.detail === "string") {
        errorMessage = errorData.detail;
      } else if (Array.isArray(errorData.detail)) {
        errorMessage = errorData.detail
          .map((err: any) => `${err.loc.join(".")}: ${err.msg}`)
          .join(", ");
      } else {
        errorMessage = JSON.stringify(errorData.detail);
      }
    }
    
    throw new Error(errorMessage);
  }

  return response.json();
}

export async function getReport(requestId: string): Promise<PredictionResponse> {
  const response = await fetch(`${API_BASE_URL}/report/${requestId}`);
  if (!response.ok) {
    throw new Error("Failed to fetch report");
  }
  return response.json();
}

export interface ChatRequest {
  request_id: string;
  message: string;
  report_data: PredictionResponse;  // Full report sent so backend never needs to look it up
}

export interface ChatResponse {
  answer: string;
  model_source: string;
}

export async function chatWithReport(data: ChatRequest): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/report/${data.request_id}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: data.message, report_data: data.report_data }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    const error = new Error(errorData.detail || "Failed to connect to AI chat") as any;
    error.status = response.status;
    throw error;
  }

  return response.json();
}

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/predict/health`);
    return response.ok;
  } catch (error) {
    console.error("Backend warm-up failed:", error);
    return false;
  }
}
