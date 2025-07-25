import React, { useState, useCallback, useEffect } from "react";
import { Editor } from "@monaco-editor/react";
import { Loader2, Code, MessageSquare, Zap, Upload, GitBranch, GitCommit } from "lucide-react";
import { motion } from "framer-motion";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const SAMPLE_CODE: Record<string, string> = {
  python: `def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
    
    def insert(self, value):
        if value < self.value:
            if self.left is None:
                self.left = BinaryTree(value)
            else:
                self.left.insert(value)
        else:
            if self.right is None:
                self.right = BinaryTree(value)
            else:
                self.right.insert(value)`,
  javascript: `function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

class EventEmitter {
    constructor() {
        this.events = {};
    }
    
    on(event, listener) {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event].push(listener);
    }
    
    emit(event, ...args) {
        if (this.events[event]) {
            this.events[event].forEach(listener => listener.apply(this, args));
        }
    }
}`
};

type Language = "python" | "javascript";

const Assistant: React.FC = () => {
  const [code, setCode] = useState<string>(SAMPLE_CODE.python);
  const [language, setLanguage] = useState<Language>("python");
  const [comments, setComments] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [confidence, setConfidence] = useState(0);
  const [processingTime, setProcessingTime] = useState(0);

  const [activeTab, setActiveTab] = useState<"comments" | "commit-msg">("comments");
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [diffText, setDiffText] = useState("");
  const [commitMessage, setCommitMessage] = useState("");
  const [commitConfidence, setCommitConfidence] = useState(0);
  const [isGeneratingDiff, setIsGeneratingDiff] = useState(false);
  const [isGeneratingCommit, setIsGeneratingCommit] = useState(false);
  const [beforeCode, setBeforeCode] = useState("");
  const [afterCode, setAfterCode] = useState("");
  const [inputMethod, setInputMethod] = useState<"file" | "code">("file");

  const generateComments = useCallback(async () => {
    if (!code.trim()) return;
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/generate-comments`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          code: code,
          language: language,
          style: "docstring"
        })
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setComments(data.comments);
      setConfidence(data.confidence);
      setProcessingTime(data.processing_time);
    } catch (error) {
      setComments(["Error: Unable to generate comments. Please try again."]);
    } finally {
      setLoading(false);
    }
  }, [code, language]);

  const handleLanguageChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newLanguage = e.target.value as Language;
    setLanguage(newLanguage);
    setCode(SAMPLE_CODE[newLanguage]);
    setComments([]);
  };

  const handleEditorChange = (value?: string) => {
    setCode(value || "");
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    setUploadedFiles(files);
  };

  const generateDiff = async () => {
    setIsGeneratingDiff(true);
    try {
      if (inputMethod === "file") {
        if (uploadedFiles.length < 2) return;
        const formData = new FormData();
        uploadedFiles.forEach(file => {
          formData.append("files", file);
        });
        const response = await fetch(`${API_BASE_URL}/generate-diff`, {
          method: "POST",
          body: formData,
        });
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setDiffText(data.diff);
      } else {
        // Generate diff from code inputs
        if (!beforeCode.trim() || !afterCode.trim()) return;
        
        // Simple diff generation (you can enhance this)
        const beforeLines = beforeCode.split('\n');
        const afterLines = afterCode.split('\n');
        
        let diff = `--- Before\n+++ After\n`;
        const maxLines = Math.max(beforeLines.length, afterLines.length);
        
        for (let i = 0; i < maxLines; i++) {
          const beforeLine = beforeLines[i] || '';
          const afterLine = afterLines[i] || '';
          
          if (beforeLine !== afterLine) {
            if (beforeLine) diff += `- ${beforeLine}\n`;
            if (afterLine) diff += `+ ${afterLine}\n`;
          } else if (beforeLine) {
            diff += `  ${beforeLine}\n`;
          }
        }
        
        setDiffText(diff);
      }
    } catch (error) {
      setDiffText("Error: Unable to generate diff. Please try again.");
    } finally {
      setIsGeneratingDiff(false);
    }
  };

  const generateCommitMessage = async () => {
    if (!diffText.trim()) return;
    setIsGeneratingCommit(true);
    try {
      const response = await fetch(`${API_BASE_URL}/generate-commit-msg`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          diff: diffText,
        }),
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setCommitMessage(data.message);
      setCommitConfidence(data.confidence);
    } catch (error) {
      setCommitMessage("Error: Unable to generate commit message. Please try again.");
    } finally {
      setIsGeneratingCommit(false);
    }
  };



  useEffect(() => {
    const timer = setTimeout(() => {
      if (code.trim() && code !== SAMPLE_CODE[language]) {
        generateComments();
      }
    }, 1500);
    return () => clearTimeout(timer);
  }, [code, generateComments, language]);

  return (
    <div className="min-h-screen pt-20 pb-16 bg-gradient-to-br from-gray-900 to-gray-800 text-white">
      <div className="max-w-5xl mx-auto px-4 py-8">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              Claso - AI Code Assistant
            </h1>
            <p className="text-gray-300 text-lg">
              Generate intelligent comments and commit messages using our custom-trained transformer model
            </p>
          </div>

          <div className="flex justify-center mb-8">
            <div className="flex space-x-4 bg-gray-800 p-2 rounded-lg">
              <button
                onClick={() => setActiveTab("comments")}
                className={`px-6 py-2 rounded-lg transition-colors ${
                  activeTab === "comments"
                    ? "bg-blue-600 text-white"
                    : "text-gray-300 hover:text-white"
                }`}
              >
                Code Comments
              </button>
              <button
                onClick={() => setActiveTab("commit-msg")}
                className={`px-6 py-2 rounded-lg transition-colors ${
                  activeTab === "commit-msg"
                    ? "bg-blue-600 text-white"
                    : "text-gray-300 hover:text-white"
                }`}
              >
                Commit Messages
              </button>
            </div>
          </div>

          {activeTab === "comments" && (
            <div className="space-y-8">
              {/* Code Editor Panel - Full Width */}
              <div className="bg-gray-800 border border-gray-700 rounded-lg">
                <div className="flex items-center justify-between p-4 border-b border-gray-700">
                  <div className="flex items-center gap-2">
                    <Code className="w-5 h-5" />
                    <span>Code Editor</span>
                  </div>
                  <div className="flex items-center gap-4">
                    <select
                      value={language}
                      onChange={handleLanguageChange}
                      className="w-32 bg-gray-700 border-gray-600 rounded px-2 py-1 text-white"
                    >
                      <option value="python">Python</option>
                      <option value="javascript">JavaScript</option>
                    </select>
                    <button
                      onClick={generateComments}
                      disabled={loading || !code.trim()}
                      className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded text-white flex items-center"
                    >
                      {loading ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Generating...
                        </>
                      ) : (
                        <>
                          <Zap className="w-4 h-4 mr-2" /> Generate
                        </>
                      )}
                    </button>
                  </div>
                </div>
                <div className="p-4">
                  <div className="h-[600px] border border-gray-600 rounded-lg overflow-hidden">
                    <Editor
                      height="100%"
                      language={language}
                      value={code}
                      onChange={handleEditorChange}
                      theme="vs-dark"
                      options={{
                        minimap: { enabled: true },
                        fontSize: 14,
                        lineNumbers: "on",
                        roundedSelection: false,
                        scrollBeyondLastLine: false,
                        automaticLayout: true,
                        wordWrap: "on",
                      }}
                    />
                  </div>
                </div>
              </div>

              {/* Comments Panel - Below Editor */}
              <div className="bg-gray-800 border border-gray-700 rounded-lg">
                <div className="flex items-center justify-between p-4 border-b border-gray-700">
                  <div className="flex items-center gap-2">
                    <MessageSquare className="w-5 h-5" />
                    <span>Generated Comments</span>
                  </div>
                  {confidence > 0 && (
                    <div className="flex items-center gap-2">
                      <span className="bg-green-600 text-xs px-2 py-1 rounded">{Math.round(confidence * 100)}% confident</span>
                      <span className="text-gray-300 text-xs">{processingTime.toFixed(2)}s</span>
                    </div>
                  )}
                </div>
                <div className="p-4 space-y-4">
                  {loading ? (
                    <div className="flex items-center justify-center h-32">
                      <div className="text-center">
                        <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4 text-blue-400" />
                        <p className="text-gray-400">Analyzing your code...</p>
                      </div>
                    </div>
                  ) : comments.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {comments.map((comment, index) => (
                        <div
                          key={index}
                          className="bg-gray-700 p-4 rounded-lg border border-gray-600 hover:border-gray-500 transition-colors"
                        >
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs text-gray-300">Suggestion {index + 1}</span>
                          </div>
                          <pre className="text-sm text-gray-200 whitespace-pre-wrap font-mono">{comment}</pre>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center h-32 flex items-center justify-center">
                      <div className="text-gray-400">
                        <MessageSquare className="w-12 h-12 mx-auto mb-4 opacity-50" />
                        <p>Paste your code and click "Generate" to see AI-powered comments</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {activeTab === "commit-msg" && (
            <div className="space-y-8">
              {/* Input Method Selection */}
              <div className="flex justify-center">
                <div className="flex space-x-4 bg-gray-800 p-2 rounded-lg">
                  <button
                    onClick={() => setInputMethod("file")}
                    className={`px-4 py-2 rounded-lg transition-colors ${
                      inputMethod === "file"
                        ? "bg-blue-600 text-white"
                        : "text-gray-300 hover:text-white"
                    }`}
                  >
                    <Upload className="w-4 h-4 inline mr-2" />
                    Upload Files
                  </button>
                  <button
                    onClick={() => setInputMethod("code")}
                    className={`px-4 py-2 rounded-lg transition-colors ${
                      inputMethod === "code"
                        ? "bg-blue-600 text-white"
                        : "text-gray-300 hover:text-white"
                    }`}
                  >
                    <Code className="w-4 h-4 inline mr-2" />
                    Paste Code
                  </button>
                </div>
              </div>

              {inputMethod === "file" ? (
                /* File Upload Section */
                <div className="bg-gray-800 border border-gray-700 rounded-lg">
                  <div className="flex items-center justify-between p-4 border-b border-gray-700">
                    <div className="flex items-center gap-2">
                      <Upload className="w-5 h-5" />
                      <span>File Upload (Before & After)</span>
                    </div>
                    <button
                      onClick={generateDiff}
                      disabled={isGeneratingDiff || uploadedFiles.length < 2}
                      className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 px-4 py-2 rounded text-white flex items-center"
                    >
                      {isGeneratingDiff ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Generating Diff...
                        </>
                      ) : (
                        <>
                          <GitBranch className="w-4 h-4 mr-2" /> Generate Diff
                        </>
                      )}
                    </button>
                  </div>
                  <div className="p-6 space-y-6">
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-3">
                        Upload Files (Select 2 files: before and after versions)
                      </label>
                      <input
                        type="file"
                        multiple
                        onChange={handleFileUpload}
                        className="block w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700"
                      />
                    </div>
                    {uploadedFiles.length > 0 && (
                      <div>
                        <h4 className="text-sm font-medium text-gray-300 mb-3">Uploaded Files:</h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          {uploadedFiles.map((file, index) => (
                            <div key={index} className="text-sm text-gray-300 bg-gray-700 p-3 rounded-lg border border-gray-600">
                              <div className="font-medium">{index === 0 ? "Before:" : "After:"} {file.name}</div>
                              <div className="text-gray-400 text-xs mt-1">Size: {(file.size / 1024).toFixed(1)} KB</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                /* Code Input Section */
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Before Code Editor */}
                  <div className="bg-gray-800 border border-gray-700 rounded-lg">
                    <div className="flex items-center justify-between p-4 border-b border-gray-700">
                      <div className="flex items-center gap-2">
                        <Code className="w-5 h-5" />
                        <span>Before (Original Code)</span>
                      </div>
                    </div>
                    <div className="p-4">
                      <div className="h-[400px] border border-gray-600 rounded-lg overflow-hidden">
                        <Editor
                          height="100%"
                          language={language}
                          value={beforeCode}
                          onChange={(value) => setBeforeCode(value || "")}
                          theme="vs-dark"
                          options={{
                            minimap: { enabled: false },
                            fontSize: 14,
                            lineNumbers: "on",
                            roundedSelection: false,
                            scrollBeyondLastLine: false,
                            automaticLayout: true,
                            wordWrap: "on",
                          }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* After Code Editor */}
                  <div className="bg-gray-800 border border-gray-700 rounded-lg">
                    <div className="flex items-center justify-between p-4 border-b border-gray-700">
                      <div className="flex items-center gap-2">
                        <Code className="w-5 h-5" />
                        <span>After (Modified Code)</span>
                      </div>
                    </div>
                    <div className="p-4">
                      <div className="h-[400px] border border-gray-600 rounded-lg overflow-hidden">
                        <Editor
                          height="100%"
                          language={language}
                          value={afterCode}
                          onChange={(value) => setAfterCode(value || "")}
                          theme="vs-dark"
                          options={{
                            minimap: { enabled: false },
                            fontSize: 14,
                            lineNumbers: "on",
                            roundedSelection: false,
                            scrollBeyondLastLine: false,
                            automaticLayout: true,
                            wordWrap: "on",
                          }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Generate Diff Button for Code Input */}
                  <div className="lg:col-span-2 flex justify-center">
                    <button
                      onClick={generateDiff}
                      disabled={isGeneratingDiff || !beforeCode.trim() || !afterCode.trim()}
                      className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 px-6 py-3 rounded-lg text-white flex items-center"
                    >
                      {isGeneratingDiff ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Generating Diff...
                        </>
                      ) : (
                        <>
                          <GitBranch className="w-4 h-4 mr-2" /> Generate Diff
                        </>
                      )}
                    </button>
                  </div>
                </div>
              )}

              {/* Generated Diff Display */}
              {diffText && (
                <div className="bg-gray-800 border border-gray-700 rounded-lg">
                  <div className="flex items-center justify-between p-4 border-b border-gray-700">
                    <div className="flex items-center gap-2">
                      <GitBranch className="w-5 h-5" />
                      <span>Generated Diff</span>
                    </div>
                  </div>
                  <div className="p-4">
                    <pre className="text-sm text-gray-200 bg-gray-900 p-4 rounded border border-gray-600 overflow-x-auto max-h-64 overflow-y-auto">{diffText}</pre>
                  </div>
                </div>
              )}

              {/* Commit Message Panel */}
              {diffText && (
                <div className="bg-gray-800 border border-gray-700 rounded-lg">
                  <div className="flex items-center justify-between p-4 border-b border-gray-700">
                    <div className="flex items-center gap-2">
                      <GitCommit className="w-5 h-5" />
                      <span>Generated Commit Message</span>
                    </div>
                    {commitConfidence > 0 && (
                      <div className="flex items-center gap-2">
                        <span className="bg-green-600 text-xs px-2 py-1 rounded">{Math.round(commitConfidence * 100)}% confident</span>
                      </div>
                    )}
                  </div>
                  <div className="p-4 space-y-4">
                    {!isGeneratingCommit && !commitMessage && (
                      <button
                        onClick={generateCommitMessage}
                        className="w-full bg-green-600 hover:bg-green-700 px-4 py-3 rounded-lg text-white flex items-center justify-center"
                      >
                        <GitCommit className="w-4 h-4 mr-2" /> Generate Commit Message
                      </button>
                    )}
                    
                    {isGeneratingCommit ? (
                      <div className="flex items-center justify-center h-32">
                        <div className="text-center">
                          <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4 text-blue-400" />
                          <p className="text-gray-400">Generating commit message...</p>
                        </div>
                      </div>
                    ) : commitMessage ? (
                      <div className="bg-gray-700 p-4 rounded-lg border border-gray-600">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs text-gray-300">Suggested Commit Message</span>
                        </div>
                        <pre className="text-sm text-gray-200 whitespace-pre-wrap font-mono">{commitMessage}</pre>
                      </div>
                    ) : null}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Features Section */}
          <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 text-center">
              <Code className="w-12 h-12 mx-auto mb-4 text-blue-400" />
              <h3 className="text-lg font-semibold mb-2 text-white">Multi-Language Support</h3>
              <p className="text-gray-400 text-sm">
                Supports Python and JavaScript with language-specific comment styles
              </p>
            </div>
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 text-center">
              <GitCommit className="w-12 h-12 mx-auto mb-4 text-purple-400" />
              <h3 className="text-lg font-semibold mb-2 text-white">Commit Message Generation</h3>
              <p className="text-gray-400 text-sm">
                Upload before/after files to generate intelligent commit messages
              </p>
            </div>
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 text-center">
              <Zap className="w-12 h-12 mx-auto mb-4 text-green-400" />
              <h3 className="text-lg font-semibold mb-2 text-white">Real-time Generation</h3>
              <p className="text-gray-400 text-sm">
                Lightning-fast generation powered by custom transformers
              </p>
            </div>
          </div>


        </motion.div>
      </div>
    </div>
  );
};

export default Assistant; 