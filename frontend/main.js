// ============================================
// 설정
// ============================================
// API 주소: 백엔드는 항상 localhost:8000에서 실행
// 프론트엔드는 localhost:8080에서 실행
const DEFAULT_API_BASE = "http://localhost:8000";

function getApiBase() {
  // localStorage에서 저장된 값 확인
  const saved = localStorage.getItem("api_base_url");
  if (saved && saved.trim() && saved.startsWith("http")) {
    console.log(`[API] localStorage에서 주소 사용: ${saved}`);
    return saved.trim();
  }
  
  // localStorage 값이 없거나 잘못되었으면 기본값 사용
  console.log(`[API] 기본 API 주소 사용: ${DEFAULT_API_BASE}`);
  return DEFAULT_API_BASE;
}

let API_BASE = getApiBase();

// 현재 세션 정보 저장 (피드백 추적용)
let currentSession = {
  sessionId: null,
  question: null,
  answer: null,
  contexts: []
};

// API 주소 변경 함수
function setApiBase(newUrl) {
  API_BASE = newUrl;
  localStorage.setItem("api_base_url", newUrl);
  // API 상태 다시 확인
  loadApiStatusAndKeywords();
}

// ============================================
// DOM 요소 참조
// ============================================
// DOM 요소 참조 (초기화는 DOMContentLoaded에서)
let els = {};

function initElements() {
  els = {
    // 상태
  status: document.getElementById("api-status"),
    statusText: document.querySelector("#api-status .status-text"),
    statusDot: document.querySelector("#api-status .status-dot"),
    
    // 시스템 정보
    systemInfo: document.getElementById("system-info"),
    infoApiUrl: document.getElementById("info-api-url"),
    infoDevice: document.getElementById("info-device"),
    infoDocs: document.getElementById("info-docs"),
    infoKeywords: document.getElementById("info-keywords"),
    
    // 질문 입력
  question: document.getElementById("question"),
  sendBtn: document.getElementById("send-btn"),
  clearBtn: document.getElementById("clear-answer-btn"),
  infoLine: document.getElementById("info-line"),

    // 옵션
  topkInput: document.getElementById("topk-input"),
  topkLabel: document.getElementById("topk-label"),
  maxTokensInput: document.getElementById("max-tokens-input"),
  maxTokensLabel: document.getElementById("max-tokens-label"),
  keywordSelect: document.getElementById("keyword-select"),
  contextOnly: document.getElementById("context-only"),
  debugFlag: document.getElementById("debug-flag"),

    // 결과
  answer: document.getElementById("answer"),
  contexts: document.getElementById("contexts"),
  contextCount: document.getElementById("context-count"),
  
    // 피드백
  feedbackArea: document.getElementById("feedback-area"),
  feedbackLikeBtn: document.getElementById("feedback-like-btn"),
  feedbackDislikeBtn: document.getElementById("feedback-dislike-btn"),
  feedbackThanks: document.getElementById("feedback-thanks"),
    questionHistory: document.getElementById("question-history"),
    clearHistoryBtn: document.getElementById("clear-history-btn"),
    refreshDocsStatsBtn: document.getElementById("refresh-docs-stats-btn"),
};
}

// ============================================
// 상태 표시 유틸
// ============================================
function setStatusOk(text) {
  if (els.statusText) els.statusText.textContent = text;
  els.status.classList.remove("status-error");
  els.status.classList.add("status-ok");
}

function setStatusError(text) {
  if (els.statusText) els.statusText.textContent = text;
  els.status.classList.remove("status-ok");
  els.status.classList.add("status-error");
}

function setInfo(text, type = "info") {
  if (!els.infoLine) return;
  
  els.infoLine.textContent = text || "";
  els.infoLine.className = "info-line";
  
  if (type === "error") {
    els.infoLine.style.color = "var(--danger)";
  } else if (type === "success") {
    els.infoLine.style.color = "var(--success)";
  } else {
    els.infoLine.style.color = "var(--text-sub)";
  }
}

// ============================================
// 시스템 정보 표시 (MODEL_INFO 활용)
// ============================================
function renderSystemInfo(data) {
  if (!data || !els.systemInfo) return;
  
  // API 주소 표시
  if (els.infoApiUrl) {
    els.infoApiUrl.textContent = API_BASE;
  }
  
  // num_docs가 있으면 우선 사용 (keywords 합산은 중복 카운트이므로 사용하지 않음)
  const numDocs = data.num_docs !== undefined && data.num_docs > 0
    ? data.num_docs 
    : 0;
  
  // keyword_list가 있으면 우선 사용, 없으면 keywords에서 추출
  const keywordList = data.keyword_list && data.keyword_list.length > 0
    ? data.keyword_list
    : Object.keys(data.keywords || {}).filter(k => k && k !== "unknown");
  
  // Device 정보 (data에서 가져오기)
  const device = data.device || "unknown";
  if (els.infoDevice) {
    els.infoDevice.textContent = device;
  }
  
  if (els.infoDocs) {
    els.infoDocs.textContent = `${numDocs}개`;
  }
  
  if (els.infoKeywords) {
    els.infoKeywords.textContent = keywordList.length > 0 
      ? `${keywordList.length}개 (${keywordList.slice(0, 3).join(", ")}${keywordList.length > 3 ? "..." : ""})`
      : "없음";
  }
  
  els.systemInfo.style.display = "flex";
}

// ============================================
// API 상태 및 키워드 로딩
// ============================================
async function loadApiStatusAndKeywords() {
  // API_BASE가 제대로 설정되었는지 확인
  if (!API_BASE || !API_BASE.startsWith('http')) {
    console.error(`[API] 잘못된 API 주소: ${API_BASE}`);
    API_BASE = DEFAULT_API_BASE;
    console.log(`[API] 기본값으로 재설정: ${API_BASE}`);
  }
  
  try {
    console.log(`[API] 연결 시도: ${API_BASE}/`);
    const res = await fetch(`${API_BASE}/`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      // CORS 문제 해결을 위한 옵션
      mode: 'cors',
      cache: 'no-cache'
    });
    
    console.log(`[API] 응답 상태: ${res.status} ${res.statusText}`);
    
    if (!res.ok) {
      const errorText = await res.text();
      console.error(`[API] 응답 실패: ${res.status}`, errorText);
      setStatusError(`API 응답 실패: ${res.status}`);
      setInfo(`서버 응답 오류: ${res.status} ${res.statusText}`, "error");
      return;
    }
    
    const data = await res.json();
    console.log('[API] 응답 성공:', data);
    setStatusOk("API 연결 성공");
    setInfo("", "success");
    
    // 시스템 정보 표시
    renderSystemInfo(data);

    // 키워드 정보 처리
    if (data.keywords && Object.keys(data.keywords).length > 0) {
      // 키워드 통계는 문서 통계에 통합됨
      fillKeywordSelect(Object.keys(data.keywords));
    } else {
      // /keywords 엔드포인트로 재시도
      await loadKeywordsFallback();
    }
  } catch (err) {
    console.error("[loadApiStatusAndKeywords 오류]", err);
    console.error("[API] 요청 URL:", `${API_BASE}/`);
    console.error("[API] 오류 타입:", err.name);
    console.error("[API] 오류 메시지:", err.message);
    
    let errorMsg = "서버에 연결할 수 없습니다";
    if (err.name === "TypeError" && err.message.includes("Failed to fetch")) {
      errorMsg = `연결 실패: ${API_BASE}에 접속할 수 없습니다. 서버가 실행 중인지 확인하세요.`;
    } else if (err.message) {
      errorMsg = err.message;
    }
    
    setStatusError("API 연결 실패");
    setInfo(`연결 오류: ${errorMsg}`, "error");
    
  }
}

async function loadKeywordsFallback() {
  try {
    const res = await fetch(`${API_BASE}/keywords`);
    if (!res.ok) {
      console.warn("[keywords fallback] 응답 실패:", res.status);
      return;
    }
    
    const stats = await res.json();
    if (stats && !stats.error) {
    // 키워드 통계는 문서 통계에 통합됨
    fillKeywordSelect(Object.keys(stats));
    } else if (stats.error) {
      console.error("[keywords] 오류:", stats.error);
    }
  } catch (err) {
    console.error("[keywords fallback] 실패", err);
  }
}

function fillKeywordSelect(keywordList) {
  if (!els.keywordSelect) return;
  
  els.keywordSelect.innerHTML = '<option value="">(전체 키워드)</option>';

  keywordList
    .filter((kw) => kw && kw !== "unknown")
    .sort()
    .forEach((kw) => {
      const opt = document.createElement("option");
      opt.value = kw;
      opt.textContent = kw;
      els.keywordSelect.appendChild(opt);
    });
}


// ============================================
// 슬라이더 라벨 동기화 (init 함수로 이동됨)
// ============================================

// ============================================
// 질문 전송 및 답변 처리
// ============================================
async function sendQuestion() {
  const question = els.question?.value.trim() || "";
  const top_k = parseInt(els.topkInput?.value, 10) || 3;
  const max_new_tokens = parseInt(els.maxTokensInput?.value, 10) || 256;
  const filter_keyword = els.keywordSelect?.value || null;
  const context_only = els.contextOnly?.checked || false;
  const debug = els.debugFlag?.checked || false;

  if (!question) {
    setInfo("질문을 입력해주세요!", "error");
    els.question?.focus();
    return;
  }

  // UI 상태 변경
  setInfo("🤔 모델이 생각 중입니다... (GPU: 빠름, CPU: 1-2분 소요)", "info");
  
  if (els.answer) {
    els.answer.innerHTML = `
      <div class="loading-state">
        <div class="spinner"></div>
        <p>답변을 생성하는 중입니다...</p>
      </div>
    `;
  }
  
  if (els.contexts) {
    els.contexts.innerHTML = `
      <div class="loading-state">
        <div class="spinner"></div>
        <p>컨텍스트를 검색하는 중입니다...</p>
      </div>
    `;
  }
  
  if (els.contextCount) {
    els.contextCount.textContent = "검색 중...";
  }
  
  if (els.sendBtn) {
  els.sendBtn.disabled = true;
    const btnText = els.sendBtn.querySelector(".btn-text");
    if (btnText) {
      btnText.textContent = "생성 중...";
    }
  }
  
  // API_BASE가 제대로 설정되었는지 확인
  if (!API_BASE || !API_BASE.startsWith('http')) {
    console.error(`[API] 잘못된 API 주소: ${API_BASE}`);
    API_BASE = DEFAULT_API_BASE;
    console.log(`[API] 기본값으로 재설정: ${API_BASE}`);
  }
  
  try {
    const startTime = Date.now();
    
    console.log(`[API] 질문 전송: ${API_BASE}/chat`);
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      mode: 'cors',
      cache: 'no-cache',
      body: JSON.stringify({
        question,
        top_k,
        max_new_tokens,
        filter_keyword,
        context_only,
        debug,
      }),
    });
    
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`[API] 응답 상태: ${res.status} ${res.statusText} (${elapsed}s)`);

    if (!res.ok) {
      const errorText = await res.text();
      console.error(`[API] 응답 실패: ${res.status}`, errorText.substring(0, 200));
      
      // 501 오류는 잘못된 서버(정적 파일 서버)에 요청이 간 경우
      if (res.status === 501 || (errorText.includes("Unsupported method") && errorText.includes("POST"))) {
        setInfo(`잘못된 서버에 연결되었습니다 (501). API 주소를 확인하세요.`, "error");
        setStatusError("서버 주소 오류");
        
        if (els.answer) {
          els.answer.innerHTML = `
            <div class="empty-state">
              <div class="empty-icon">❌</div>
              <p><strong>잘못된 서버에 연결되었습니다</strong><br/>상태 코드: ${res.status}</p>
              <p style="margin-top: 12px; font-size: 0.9em; color: var(--text-sub);">
                현재 API 주소: <code style="background: var(--bg-secondary); padding: 2px 6px; border-radius: 3px;">${API_BASE}</code><br/><br/>
                <strong>해결 방법:</strong><br/>
                1. 시스템 정보의 "API 주소"를 클릭하여 변경<br/>
                2. 올바른 주소: <code style="background: var(--bg-secondary); padding: 2px 6px; border-radius: 3px;">http://localhost:8000</code>
              </p>
            </div>
          `;
        }
      } else {
        setInfo(`API 요청 실패 (${res.status}): ${errorText.substring(0, 100)}`, "error");
      setStatusError("요청 실패");
        
        if (els.answer) {
          els.answer.innerHTML = `
            <div class="empty-state">
              <div class="empty-icon">❌</div>
              <p>서버 오류가 발생했습니다.<br/>상태 코드: ${res.status}</p>
            </div>
          `;
        }
      }
      
      if (els.contexts) {
        els.contexts.innerHTML = `
          <div class="empty-state">
            <div class="empty-icon">⚠️</div>
            <p>컨텍스트를 불러올 수 없습니다.</p>
          </div>
        `;
      }
      return;
    }

    const data = await res.json();
    setInfo(`✅ 응답 완료 (${elapsed}초)`, "success");

    // 신뢰도 점수 표시 (헤더 옆에 작게)
    const confidenceBadge = document.getElementById("confidence-badge");
    const confidenceHelp = document.getElementById("confidence-help");
    if (confidenceBadge) {
      if (data.confidence !== null && data.confidence !== undefined) {
        const confidencePercent = Math.round(data.confidence * 100);
        let confidenceColor = "var(--danger)";
        let confidenceLabel = "낮음";
        
        if (data.confidence >= 0.7) {
          confidenceColor = "var(--success)";
          confidenceLabel = "높음";
        } else if (data.confidence >= 0.5) {
          confidenceColor = "var(--warning)";
          confidenceLabel = "보통";
        }
        
        confidenceBadge.textContent = `신뢰도: ${confidenceLabel} (${confidencePercent}%)`;
        confidenceBadge.style.color = confidenceColor;
        confidenceBadge.style.borderLeft = `2px solid ${confidenceColor}`;
        confidenceBadge.style.display = "inline-block";
        
        // 물음표 아이콘도 표시
        if (confidenceHelp) {
          confidenceHelp.style.display = "inline-block";
        }
      } else {
        confidenceBadge.style.display = "none";
        if (confidenceHelp) {
          confidenceHelp.style.display = "none";
        }
      }
    }

    // 답변 표시
    if (els.answer) {
    if (data.answer) {
        let answerHtml = "";
        
        if (context_only && data.answer.includes("컨텍스트만")) {
          answerHtml += `
            <div style="padding: 12px; background: var(--accent-soft); border-radius: var(--radius-md); margin-bottom: 12px;">
              <strong>ℹ️ 컨텍스트 전용 모드</strong><br/>
              답변은 생성되지 않았습니다. 아래 컨텍스트를 확인하세요.
            </div>
            <div style="white-space: pre-wrap;">${data.answer}</div>
          `;
        } else {
          answerHtml += `<div style="white-space: pre-wrap;">${data.answer}</div>`;
        }
        
        els.answer.innerHTML = answerHtml;
    } else {
        els.answer.innerHTML = `
          <div class="empty-state">
            <div class="empty-icon">⚠️</div>
            <p>답변이 비어 있습니다.<br/>백엔드 로그를 확인하세요.</p>
          </div>
        `;
      }
    }

    // 컨텍스트 표시
    renderContexts(data.contexts || [], data.used_keyword || null);
    
    // 관련 질문 표시
    renderRelatedQuestions(data.related_questions || []);
    
    // 질문 히스토리에 추가
    addToHistory(question, data.answer || "", data.contexts || []);
    
    // 세션 정보 저장 및 피드백 영역 표시
    currentSession = {
      sessionId: data.session_id || null,
      question: question,
      answer: data.answer || "",
      contexts: data.contexts || []
    };
    
    // 피드백 영역 표시
    if (els.feedbackArea && currentSession.sessionId) {
      els.feedbackArea.style.display = "block";
      if (els.feedbackThanks) {
        els.feedbackThanks.style.display = "none";
      }
      // 피드백 버튼 다시 활성화
      if (els.feedbackLikeBtn) els.feedbackLikeBtn.disabled = false;
      if (els.feedbackDislikeBtn) els.feedbackDislikeBtn.disabled = false;
    }
    
  } catch (err) {
    console.error("[sendQuestion 오류]", err);
    console.error("[API] 요청 URL:", `${API_BASE}/chat`);
    console.error("[API] 오류 타입:", err.name);
    console.error("[API] 오류 메시지:", err.message);
    
    let errorMsg = "알 수 없는 오류";
    if (err.name === "TypeError" && err.message.includes("Failed to fetch")) {
      errorMsg = `연결 실패: ${API_BASE}에 접속할 수 없습니다. 서버가 실행 중인지 확인하세요.`;
    } else if (err.message) {
      errorMsg = err.message;
    }
    
    setInfo(`❌ 요청 중 오류: ${errorMsg}`, "error");
    setStatusError("에러 발생");
    
    if (els.answer) {
      els.answer.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">❌</div>
          <p>오류가 발생했습니다:<br/>${errorMsg}</p>
          <p style="margin-top: 8px; font-size: 0.85em; color: var(--text-sub);">
            API 주소: ${API_BASE}
          </p>
        </div>
      `;
    }
    
    if (els.contexts) {
      els.contexts.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">⚠️</div>
          <p>컨텍스트를 불러올 수 없습니다.</p>
        </div>
      `;
    }
  } finally {
    if (els.sendBtn) {
    els.sendBtn.disabled = false;
      const btnText = els.sendBtn.querySelector(".btn-text");
      if (btnText) {
        btnText.textContent = "질문 보내기";
      }
    }
  }
}

function renderContexts(contexts, usedKeyword = null) {
  if (!els.contexts) return;
  
  els.contexts.innerHTML = "";
  
  if (els.contextCount) {
  els.contextCount.textContent = `${contexts.length} 개`;
  }

  if (!contexts.length) {
    els.contexts.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">📄</div>
        <p>검색된 컨텍스트가 없습니다.<br/>질문을 더 구체적으로 바꿔보세요.</p>
      </div>
    `;
    return;
  }

  const list = document.createElement("div");
  list.className = "context-cards";

  // 사용된 키워드 정보 표시
  if (usedKeyword) {
    const filterInfo = document.createElement("div");
    filterInfo.className = "context-filter-info";
    filterInfo.innerHTML = `🔍 <strong>필터 적용:</strong> ${usedKeyword}`;
    list.appendChild(filterInfo);
  }
  
  // 컨텍스트 카드 생성
  contexts.forEach((c, idx) => {
    const card = document.createElement("article");
    card.className = "context-card";

    const score = typeof c.score === "number" ? c.score.toFixed(3) : "N/A";
    const kw = c.keyword || "";
    
    // 점수에 따라 색상 클래스 결정
    let scoreClass = "score-low";
    if (typeof c.score === "number") {
      if (c.score > 0.8) scoreClass = "score-high";
      else if (c.score > 0.6) scoreClass = "score-medium";
    }

    card.innerHTML = `
      <header>
        <span class="ctx-index">#${idx + 1}</span>
        <span class="ctx-score ${scoreClass}">score=${score}</span>
        ${kw ? `<span class="ctx-keyword">${kw}</span>` : ""}
      </header>
      <p class="ctx-text">${c.text}</p>
    `;
    
    list.appendChild(card);
  });

  els.contexts.appendChild(list);
}

// 관련 질문 표시
function renderRelatedQuestions(questions) {
  const container = document.getElementById("related-questions");
  if (!container) return;
  
  if (!questions || questions.length === 0) {
    container.style.display = "none";
    return;
  }
  
  container.style.display = "block";
  container.innerHTML = "";
  
  const title = document.createElement("div");
  title.style.cssText = "font-weight: 600; color: var(--text-main); margin-bottom: 8px; font-size: 14px;";
  title.textContent = "💡 관련 질문:";
  container.appendChild(title);
  
  const list = document.createElement("div");
  list.style.cssText = "display: flex; flex-direction: column; gap: 6px;";
  
  questions.forEach((q) => {
    const btn = document.createElement("button");
    btn.className = "related-question-btn";
    btn.textContent = q;
    btn.style.cssText = `
      padding: 8px 12px;
      background: var(--bg-input);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      text-align: left;
      cursor: pointer;
      transition: var(--transition);
      font-size: 13px;
      color: var(--text-main);
    `;
    
    btn.addEventListener("mouseenter", () => {
      btn.style.background = "var(--accent-soft)";
      btn.style.borderColor = "var(--accent)";
    });
    
    btn.addEventListener("mouseleave", () => {
      btn.style.background = "var(--bg-input)";
      btn.style.borderColor = "var(--border-subtle)";
    });
    
    btn.addEventListener("click", () => {
      if (els.question) {
        els.question.value = q;
        sendQuestion();
      }
    });
    
    list.appendChild(btn);
  });
  
  container.appendChild(list);
}

// ============================================
// 이벤트 바인딩 (init 함수로 이동됨)
// ============================================

// ============================================
// 초기화
// ============================================
function init() {
  initElements();
  loadApiStatusAndKeywords();
  
  // 슬라이더 라벨 동기화
  if (els.topkInput && els.topkLabel) {
    els.topkInput.addEventListener("input", () => {
      els.topkLabel.textContent = els.topkInput.value;
    });
  }
  
  if (els.maxTokensInput && els.maxTokensLabel) {
    els.maxTokensInput.addEventListener("input", () => {
      els.maxTokensLabel.textContent = els.maxTokensInput.value;
    });
  }
  
  // 이벤트 바인딩
  if (els.sendBtn) {
els.sendBtn.addEventListener("click", sendQuestion);
  }

  if (els.question) {
els.question.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    sendQuestion();
  }
});
  }

  if (els.clearBtn) {
els.clearBtn.addEventListener("click", () => {
      if (els.answer) {
        els.answer.innerHTML = `
          <div class="empty-state">
            <div class="empty-icon">💭</div>
            <p>왼쪽에 질문을 입력하고 전송 버튼을 눌러주세요.</p>
          </div>
        `;
      }
      
      if (els.contexts) {
        els.contexts.innerHTML = `
          <div class="empty-state">
            <div class="empty-icon">📄</div>
            <p>검색된 컨텍스트가 여기에 표시됩니다.</p>
          </div>
        `;
      }
      
      if (els.contextCount) {
  els.contextCount.textContent = "0 개";
      }
      
      // 신뢰도 배지 및 물음표 아이콘 숨기기
      const confidenceBadge = document.getElementById("confidence-badge");
      const confidenceHelp = document.getElementById("confidence-help");
      const confidenceTooltip = document.getElementById("confidence-tooltip");
      if (confidenceBadge) {
        confidenceBadge.style.display = "none";
      }
      if (confidenceHelp) {
        confidenceHelp.style.display = "none";
      }
      if (confidenceTooltip) {
        confidenceTooltip.style.display = "none";
      }
      
  setInfo("");
});
  }
  
  // refreshStatsBtn은 제거됨 (문서 통계에 통합)
  
  // API 주소 클릭 시 변경 가능
  if (els.infoApiUrl) {
    els.infoApiUrl.addEventListener("click", () => {
      const currentUrl = API_BASE;
      const newUrl = prompt(
        "API 서버 주소를 입력하세요:\n\n" +
        "기본값: http://localhost:8000\n\n" +
        "예시:\n" +
        "- http://localhost:8000 (로컬)\n" +
        "- http://192.168.0.22:8000 (네트워크 IP)",
        currentUrl
      );
      
      if (newUrl && newUrl.trim() && newUrl !== currentUrl) {
        const trimmedUrl = newUrl.trim();
        // 간단한 URL 검증
        if (trimmedUrl.startsWith("http://") || trimmedUrl.startsWith("https://")) {
          setApiBase(trimmedUrl);
          setInfo(`API 주소가 변경되었습니다: ${trimmedUrl}`, "success");
        } else {
          alert("올바른 URL 형식이 아닙니다. http:// 또는 https://로 시작해야 합니다.");
        }
      }
    });
  }
}

// ============================================
// 문서 관리 기능
// ============================================

// 탭 전환
function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");
  
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetTab = btn.dataset.tab;
      
      // 모든 탭 비활성화
      tabBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => {
        c.classList.remove("active");
        c.style.display = "none";
      });
      
      // 선택한 탭 활성화
      btn.classList.add("active");
      const targetContent = document.getElementById(`tab-${targetTab}`);
      if (targetContent) {
        targetContent.classList.add("active");
        targetContent.style.display = "block";
      }
      
      // 문서 관리 탭으로 전환 시 통계 로드
      if (targetTab === "docs") {
        loadDocsStats();
      }
    });
  });
}

// 문서 통계 로드 (키워드 통계 포함)
async function loadDocsStats() {
  const container = document.getElementById("docs-stats-content");
  if (!container) return;
  
  container.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>통계를 불러오는 중...</p></div>';
  
  try {
    const res = await fetch(`${API_BASE}/docs/stats`);
    const data = await res.json();
    
    if (data.success) {
      const stats = data.stats || {};
      const total = data.total_docs || 0;
      
      // 총 문서 수와 키워드 개수 표시
      const keywordCount = Object.keys(stats).length;
      let html = `
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; margin-bottom: 16px;">
          <div style="padding: 10px 12px; background: var(--bg-input); border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <div style="font-size: 11px; color: var(--text-sub); margin-bottom: 2px;">총 문서 수</div>
            <div style="font-size: 22px; font-weight: 700; color: var(--accent);">${total}</div>
          </div>
          <div style="padding: 10px 12px; background: var(--bg-input); border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <div style="font-size: 11px; color: var(--text-sub); margin-bottom: 2px;">키워드 종류</div>
            <div style="font-size: 22px; font-weight: 700; color: var(--accent);">${keywordCount}</div>
          </div>
        </div>
      `;
      
      // 키워드별 통계
      html += '<div style="margin-bottom: 10px;"><strong style="color: var(--text-main); font-size: 14px;">키워드별 문서 개수</strong></div>';
      html += '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 8px;">';
      
      for (const [kw, count] of Object.entries(stats).sort((a, b) => b[1] - a[1])) {
        html += `
          <div style="padding: 8px 10px; background: var(--bg-input); border-radius: var(--radius-md); border: 1px solid var(--border-subtle); transition: var(--transition);">
            <div style="font-weight: 600; color: var(--accent); font-size: 12px; margin-bottom: 2px;">${kw}</div>
            <div style="font-size: 18px; font-weight: 700; color: var(--text-main);">${count}</div>
          </div>
        `;
      }
      
      html += '</div>';
      container.innerHTML = html;
    } else {
      container.innerHTML = `<div class="docs-result error">오류: ${data.error || "알 수 없는 오류"}</div>`;
    }
  } catch (err) {
    container.innerHTML = `<div class="docs-result error">연결 오류: ${err.message}</div>`;
  }
}

// 문서 추가
function initDocsAdd() {
  const form = document.getElementById("docs-add-form");
  const result = document.getElementById("docs-add-result");
  
  if (!form || !result) return;
  
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    
    const keyword = document.getElementById("add-keyword").value.trim();
    const text = document.getElementById("add-text").value.trim();
    
    if (!keyword || !text) {
      result.innerHTML = '<div class="docs-result error">키워드와 문장을 모두 입력하세요.</div>';
      return;
    }
    
    result.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>추가 중...</p></div>';
    
    try {
      const formData = new FormData();
      formData.append("keyword", keyword);
      formData.append("text", text);
      
      const res = await fetch(`${API_BASE}/docs/add`, {
        method: "POST",
        body: formData
      });
      
      const data = await res.json();
      
      if (data.success) {
        result.innerHTML = `<div class="docs-result success">${data.message || "문서가 추가되었습니다."}</div>`;
        form.reset();
        loadDocsStats();
        loadApiStatusAndKeywords(); // 키워드 목록 업데이트
      } else {
        result.innerHTML = `<div class="docs-result error">오류: ${data.error || "알 수 없는 오류"}</div>`;
      }
    } catch (err) {
      result.innerHTML = `<div class="docs-result error">연결 오류: ${err.message}</div>`;
    }
  });
} 

// 문서 추출
function initDocsExtract() {
  const form = document.getElementById("docs-extract-form");
  const result = document.getElementById("docs-extract-result");
  
  if (!form || !result) return;
  
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById("extract-file");
    const keywords = document.getElementById("extract-keywords").value.trim();
    const fileType = document.getElementById("extract-type").value;
    
    if (!fileInput.files || !fileInput.files[0] || !keywords) {
      result.innerHTML = '<div class="docs-result error">파일과 키워드를 모두 입력하세요.</div>';
      return;
    }
    
    result.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>추출 중...</p></div>';
    
    try {
      const formData = new FormData();
      formData.append("file", fileInput.files[0]);
      formData.append("keywords", keywords);
      formData.append("file_type", fileType);
      
      const res = await fetch(`${API_BASE}/docs/extract`, {
        method: "POST",
        body: formData
      });
      
      const data = await res.json();
      
      if (data.success) {
        let html = `<div class="docs-result success">${data.message || "추출 완료"}</div>`;
        if (data.extracted) {
          html += '<div style="margin-top: 12px;"><strong>추출된 문장:</strong><ul style="margin-top: 8px;">';
          for (const [kw, count] of Object.entries(data.extracted)) {
            html += `<li>${kw}: ${count}개</li>`;
          }
          html += '</ul></div>';
        }
        result.innerHTML = html;
        form.reset();
        loadDocsStats();
        loadApiStatusAndKeywords();
      } else {
        result.innerHTML = `<div class="docs-result error">오류: ${data.error || "알 수 없는 오류"}</div>`;
      }
    } catch (err) {
      result.innerHTML = `<div class="docs-result error">연결 오류: ${err.message}</div>`;
    }
  });
}

// 문서 생성
function initDocsGenerate() {
  const form = document.getElementById("docs-generate-form");
  const result = document.getElementById("docs-generate-result");
  
  if (!form || !result) return;
  
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    
    const mode = document.getElementById("generate-mode").value;
    const keyword = document.getElementById("generate-keyword").value.trim();
    const count = parseInt(document.getElementById("generate-count").value);
    
    if (!keyword || count < 1) {
      result.innerHTML = '<div class="docs-result error">키워드와 개수를 올바르게 입력하세요.</div>';
      return;
    }
    
    result.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>생성 중...</p></div>';
    
    try {
      const formData = new FormData();
      formData.append("mode", mode);
      formData.append("keyword", keyword);
      formData.append("count", count.toString());
      
      const res = await fetch(`${API_BASE}/docs/generate`, {
        method: "POST",
        body: formData
      });
      
      const data = await res.json();
      
      if (data.success) {
        let html = `<div class="docs-result success">${data.message || "생성 완료"}</div>`;
        if (data.warning) {
          html += `<div style="margin-top: 8px; color: var(--warning);">${data.warning}</div>`;
        }
        if (data.items && data.items.length > 0) {
          html += '<div style="margin-top: 12px;"><strong>생성된 문장:</strong><ul style="margin-top: 8px;">';
          data.items.forEach(item => {
            html += `<li>${item.text}</li>`;
          });
          html += '</ul></div>';
        }
        result.innerHTML = html;
        form.reset();
        loadDocsStats();
loadApiStatusAndKeywords();
      } else {
        result.innerHTML = `<div class="docs-result error">오류: ${data.error || "알 수 없는 오류"}</div>`;
      }
    } catch (err) {
      result.innerHTML = `<div class="docs-result error">연결 오류: ${err.message}</div>`;
    }
  });
}

// 키워드별 그룹 보기
function initDocsGroup() {
  const btn = document.getElementById("load-group-btn");
  const container = document.getElementById("docs-group-content");
  
  if (!btn || !container) return;
  
  btn.addEventListener("click", async () => {
    container.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>문서 목록을 불러오는 중...</p></div>';
    
    try {
      const res = await fetch(`${API_BASE}/docs/group`);
      const data = await res.json();
      
      if (data.success) {
        const grouped = data.grouped || {};
        let html = `<div style="margin-bottom: 16px;"><strong>총 ${data.total_docs}개 문서, ${data.total_keywords}개 키워드</strong></div>`;
        
        for (const [kw, items] of Object.entries(grouped).sort()) {
          html += `<div class="keyword-group">`;
          html += `<h3>${kw} (${items.length}개)</h3>`;
          
          items.forEach(item => {
            html += `
              <div class="doc-item">
                <div class="doc-item-id">ID: ${item.id || "?"}</div>
                <div class="doc-item-text">${item.text || ""}</div>
              </div>
            `;
          });
          
          html += `</div>`;
        }
        
        container.innerHTML = html;
      } else {
        container.innerHTML = `<div class="docs-result error">오류: ${data.error || "알 수 없는 오류"}</div>`;
      }
    } catch (err) {
      container.innerHTML = `<div class="docs-result error">연결 오류: ${err.message}</div>`;
    }
  });
}

// 질문 히스토리 관리
function getHistory() {
  try {
    const history = localStorage.getItem("question_history");
    return history ? JSON.parse(history) : [];
  } catch (e) {
    return [];
  }
}

function saveHistory(history) {
  try {
    // 최대 20개만 저장
    const limited = history.slice(0, 20);
    localStorage.setItem("question_history", JSON.stringify(limited));
  } catch (e) {
    console.error("히스토리 저장 실패:", e);
  }
}

function addToHistory(question, answer, contexts) {
  const history = getHistory();
  const newEntry = {
    id: Date.now(),
    question: question,
    answer: answer.substring(0, 200) + (answer.length > 200 ? "..." : ""), // 답변은 200자로 제한
    contextCount: contexts.length,
    timestamp: new Date().toLocaleString("ko-KR"),
  };
  
  history.unshift(newEntry); // 최신이 위로
  saveHistory(history);
  renderHistory();
}

function renderHistory() {
  if (!els.questionHistory) return;
  
  const history = getHistory();
  
  if (history.length === 0) {
    els.questionHistory.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">💭</div>
        <p>질문을 보내면 여기에 히스토리가 저장됩니다.</p>
      </div>
    `;
    return;
  }
  
  let html = '';
  
  history.forEach((entry) => {
    html += `
      <div class="history-item" 
           onclick="const q = document.getElementById('question'); if(q) { q.value = ${JSON.stringify(entry.question)}; const btn = document.getElementById('send-btn'); if(btn) btn.click(); }">
        <div class="history-item-header">
          <div class="history-question">💬 ${entry.question}</div>
          <div class="history-badge">📄 ${entry.contextCount}</div>
        </div>
        <div class="history-timestamp">${entry.timestamp}</div>
      </div>
    `;
  });
  
  els.questionHistory.innerHTML = html;
}

function initHistory() {
  // 히스토리 초기 렌더링
  renderHistory();
  
  // 히스토리 지우기 버튼
  if (els.clearHistoryBtn) {
    els.clearHistoryBtn.addEventListener("click", () => {
      if (confirm("모든 질문 히스토리를 삭제하시겠습니까?")) {
        localStorage.removeItem("question_history");
        renderHistory();
      }
    });
  }
  
  // 문서 통계 새로고침 버튼
  if (els.refreshDocsStatsBtn) {
    els.refreshDocsStatsBtn.addEventListener("click", () => {
      loadDocsStats();
    });
  }
}

// 접기/펼치기 기능
function initCollapsible() {
  // 옵션 접기/펼치기
  const optionsCollapseBtn = document.getElementById("options-collapse-btn");
  const optionsContent = document.getElementById("options-content");
  
  if (optionsCollapseBtn && optionsContent) {
    optionsCollapseBtn.addEventListener("click", () => {
      optionsContent.classList.toggle("collapsed");
      optionsCollapseBtn.classList.toggle("collapsed");
    });
  }
  
  // 컨텍스트 접기/펼치기
  const contextCollapseBtn = document.getElementById("context-collapse-btn");
  const contextContent = document.getElementById("contexts");
  const contextPanel = contextContent?.closest(".panel-context");
  
  if (contextCollapseBtn && contextContent && contextPanel) {
    contextCollapseBtn.addEventListener("click", () => {
      const isCollapsed = contextPanel.classList.contains("collapsed");
      contextPanel.classList.toggle("collapsed");
      contextContent.classList.toggle("collapsed");
      contextCollapseBtn.classList.toggle("collapsed");
      contextCollapseBtn.textContent = isCollapsed ? "▼" : "▶";
    });
  }
}

// 피드백 전송 함수
async function submitFeedback(feedbackType) {
  if (!currentSession.sessionId) {
    console.error("[피드백] 세션 ID가 없습니다.");
    return;
  }
  
  if (!els.feedbackLikeBtn || !els.feedbackDislikeBtn) {
    return;
  }
  
  // 버튼 비활성화
  els.feedbackLikeBtn.disabled = true;
  els.feedbackDislikeBtn.disabled = true;
  
  try {
    const response = await fetch(`${API_BASE}/feedback`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        session_id: currentSession.sessionId,
        question: currentSession.question,
        answer: currentSession.answer,
        contexts: currentSession.contexts,
        feedback: feedbackType,  // "like" or "dislike"
        comment: null
      }),
    });
    
    const data = await response.json();
    
    if (data.success) {
      // 감사 메시지 표시
      if (els.feedbackThanks) {
        els.feedbackThanks.style.display = "block";
      }
      console.log("[피드백] 피드백이 저장되었습니다:", feedbackType);
    } else {
      console.error("[피드백] 피드백 저장 실패:", data.error);
      // 버튼 다시 활성화
      els.feedbackLikeBtn.disabled = false;
      els.feedbackDislikeBtn.disabled = false;
    }
  } catch (err) {
    console.error("[피드백] 피드백 전송 오류:", err);
    // 버튼 다시 활성화
    els.feedbackLikeBtn.disabled = false;
    els.feedbackDislikeBtn.disabled = false;
  }
}

// 피드백 초기화 함수
function initFeedback() {
  // 신뢰도 도움말 툴팁 기능
  const confidenceHelp = document.getElementById("confidence-help");
  const confidenceTooltip = document.getElementById("confidence-tooltip");
  
  if (confidenceHelp && confidenceTooltip) {
    // 마우스 호버 시 툴팁 표시
    confidenceHelp.addEventListener("mouseenter", () => {
      confidenceTooltip.style.display = "block";
    });
    
    confidenceHelp.addEventListener("mouseleave", () => {
      confidenceTooltip.style.display = "none";
    });
    
    // 툴팁 영역에서도 마우스가 벗어나면 숨김
    confidenceTooltip.addEventListener("mouseenter", () => {
      confidenceTooltip.style.display = "block";
    });
    
    confidenceTooltip.addEventListener("mouseleave", () => {
      confidenceTooltip.style.display = "none";
    });
    
    // 물음표 아이콘 호버 효과
    confidenceHelp.addEventListener("mouseenter", () => {
      confidenceHelp.style.opacity = "1";
    });
    
    confidenceHelp.addEventListener("mouseleave", () => {
      confidenceHelp.style.opacity = "0.6";
    });
  }
  
  if (els.feedbackLikeBtn) {
    els.feedbackLikeBtn.addEventListener("click", () => {
      submitFeedback("like");
    });
  }
  
  if (els.feedbackDislikeBtn) {
    els.feedbackDislikeBtn.addEventListener("click", () => {
      submitFeedback("dislike");
    });
  }
  
  // 답변 지우기 버튼 클릭 시 피드백 영역 숨기기
  if (els.clearBtn) {
    els.clearBtn.addEventListener("click", () => {
      if (els.feedbackArea) {
        els.feedbackArea.style.display = "none";
      }
      // 신뢰도 배지 및 물음표 아이콘 숨기기
      const confidenceBadge = document.getElementById("confidence-badge");
      const confidenceHelp = document.getElementById("confidence-help");
      const confidenceTooltip = document.getElementById("confidence-tooltip");
      if (confidenceBadge) {
        confidenceBadge.style.display = "none";
      }
      if (confidenceHelp) {
        confidenceHelp.style.display = "none";
      }
      if (confidenceTooltip) {
        confidenceTooltip.style.display = "none";
      }
      currentSession = {
        sessionId: null,
        question: null,
        answer: null,
        contexts: []
      };
    });
  }
}

// 페이지 로드 시 초기화
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    init();
    initTabs();
    initDocsAdd();
    initDocsExtract();
    initDocsGenerate();
    initDocsGroup();
    initCollapsible();
    initHistory();
    initFeedback();
  });
} else {
  init();
  initTabs();
  initDocsAdd();
  initDocsExtract();
  initDocsGenerate();
  initDocsGroup();
  initCollapsible();
  initHistory();
  initFeedback();
}
