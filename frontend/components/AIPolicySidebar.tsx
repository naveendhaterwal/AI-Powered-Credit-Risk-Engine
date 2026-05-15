"use client";

import React, { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Bot, ExternalLink, SendHorizonal, ShieldCheck, AlertTriangle } from "lucide-react";
import { PredictionResponse, chatWithReport } from "@/lib/api";
import { cn } from "@/lib/utils";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

interface AIPolicySidebarProps {
  data: PredictionResponse;
}

const AIPolicySidebar = ({ data }: AIPolicySidebarProps) => {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [chat, setChat] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content:
        "Ask me anything about this credit decision, policy compliance, FOIR/DTI impact, or next underwriting action.",
    },
  ]);


  const starterPrompts = useMemo(
    () => [
      "Why is this applicant risky?",
      "Explain policy violations in simple words",
      "What can improve approval chance?",
    ],
    []
  );

  const askAssistant = async (message: string) => {
    const text = message.trim();
    if (!text || isLoading) {
      return;
    }

    setChat((prev) => [...prev, { role: "user", content: text }]);
    setQuery("");
    setIsLoading(true);

    try {
      const response = await chatWithReport({
        request_id: data.request_id,
        message: text,
        report_data: data,  // ← full report sent with every message
      });

      const sourceLabel = response.model_source === "groq" ? "Groq" : "Fallback";
      setChat((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `${response.answer}\n\nSource: ${sourceLabel}`,
        },
      ]);
    } catch (error: any) {
      console.error("AI Chat Error:", error);
      
      let errorMessage = "Unable to get AI response right now.";
      
      if (error?.status === 404) {
        errorMessage = "The session has expired or the server restarted. Please try again in a few minutes.";
      } else if (error?.message) {
        errorMessage = `Error: ${error.message}. Please try again in a few minutes.`;
      }

      setChat((prev) => [
        ...prev,
        {
          role: "assistant",
          content: errorMessage,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const statusClass = (status: string) => {
    const normalized = status.toLowerCase();
    if (normalized === "compliant" || normalized === "passed") return "bg-green-500/20 text-green-400";
    if (normalized === "borderline" || normalized === "warning") return "bg-yellow-500/20 text-yellow-300";
    return "bg-red-500/20 text-red-400";
  };

  return (
    <aside className="glass border border-white/10 rounded-[2rem] h-full lg:sticky lg:top-6 flex flex-col overflow-hidden">
      <div className="p-8 border-b border-white/10 bg-white/[0.02]">
        <h3 className="text-xl font-bold flex items-center gap-2">
          <Bot className="w-6 h-6 text-primary" />
          Governance Control
        </h3>
        <p className="text-xs text-muted-foreground mt-2 font-medium">Underwriting arbitration & policy enforcement.</p>
        
        <div className="mt-6 p-4 rounded-2xl bg-white/5 border border-white/10">
          <div className="flex items-center justify-between mb-2">
            <div className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">Policy Risk Score</div>
            <div className="text-sm font-black text-white">{data.risk_analysis.policy_risk_score.toFixed(1)}</div>
          </div>
          <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
            <motion.div 
              initial={{ width: 0 }}
              animate={{ width: `${data.risk_analysis.policy_risk_score}%` }}
              className={cn(
                "h-full rounded-full",
                data.risk_analysis.policy_risk_score > 60 ? "bg-red-500" : data.risk_analysis.policy_risk_score > 30 ? "bg-yellow-500" : "bg-green-500"
              )}
            />
          </div>
        </div>
      </div>

      <div className="p-4 border-b border-white/10">
        <div className="text-[11px] uppercase tracking-wider font-bold text-muted-foreground mb-2">Quick Ask</div>
        <div className="flex flex-wrap gap-2">
          {starterPrompts.map((prompt) => (
            <button
              key={prompt}
              onClick={() => askAssistant(prompt)}
              className="text-xs px-3 py-1.5 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 transition-colors"
              disabled={isLoading}
            >
              {prompt}
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-col flex-1 min-h-0">
        <div className="flex-1 overflow-y-auto p-8 space-y-6">
          <section className="space-y-4">
            <div className="text-[10px] uppercase tracking-[0.2em] font-bold text-muted-foreground flex items-center justify-between border-b border-white/5 pb-2">
              <span>AI Conversation</span>
              <span className="text-primary/50 normal-case bg-primary/10 px-2 py-0.5 rounded-full font-black">Groq 3.1</span>
            </div>
            {chat.map((message, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className={cn(
                  "rounded-xl p-3 text-xs leading-relaxed whitespace-pre-wrap",
                  message.role === "user"
                    ? "bg-primary/15 border border-primary/30 ml-6"
                    : "bg-white/5 border border-white/10 mr-6"
                )}
              >
                {message.content}
              </motion.div>
            ))}
            {isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white/5 border border-white/10 mr-12 rounded-xl p-4 inline-flex items-center gap-1.5 self-start"
              >
                <div className="w-2 h-2 rounded-full bg-primary/60 animate-bounce" style={{ animationDelay: "0ms" }} />
                <div className="w-2 h-2 rounded-full bg-primary/60 animate-bounce" style={{ animationDelay: "150ms" }} />
                <div className="w-2 h-2 rounded-full bg-primary/60 animate-bounce" style={{ animationDelay: "300ms" }} />
              </motion.div>
            )}
          </section>
        </div>

        <form
          className="border-t border-white/10 p-4 bg-white/[0.01]"
          onSubmit={(e) => {
            e.preventDefault();
            askAssistant(query);
          }}
        >
          <div className="flex items-center gap-2.5">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask about this AI response..."
              className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm outline-none focus:border-primary/50 transition-all placeholder:text-muted-foreground/30 focus:bg-white/[0.07]"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !query.trim()}
              className="shrink-0 rounded-xl w-10 h-10 flex items-center justify-center bg-primary text-primary-foreground disabled:opacity-50 hover:bg-primary/90 transition-all active:scale-95 shadow-md shadow-primary/20"
            >
              <SendHorizonal className="w-4 h-4" />
            </button>
          </div>
        </form>
      </div>
    </aside>
  );
};

export default AIPolicySidebar;
