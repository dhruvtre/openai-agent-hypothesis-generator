'use client';

import React, { useState, useRef, useEffect } from 'react';

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  toolCalls?: ToolCall[];
}

interface ToolCall {
  name: string;
  description: string;
  timestamp: Date;
}

interface ChatSettings {
  domain: string;
  numHypotheses: number;
  researchIdea: string;
}

export default function SimpleChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [settings, setSettings] = useState<ChatSettings>({
    domain: 'AI for Drug Discovery',
    numHypotheses: 3,
    researchIdea: '',
  });
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!settings.researchIdea.trim() || isStreaming) return;

    const userMessage = settings.researchIdea;
    setMessages(prev => [...prev, { 
      role: 'user', 
      content: `Research Idea: ${userMessage}\nDomain: ${settings.domain}\nNumber of Hypotheses: ${settings.numHypotheses}`,
      timestamp: new Date()
    }]);
    setIsStreaming(true);

    try {
      const response = await fetch('http://localhost:8000/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'hypothesis-generator',
          messages: [{ role: 'user', content: userMessage }],
          stream: true,
          metadata: {
            domain: settings.domain,
            num_hypotheses: settings.numHypotheses,
          },
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let assistantMessage = '';
      let hasSeenToolCalls = false;

      if (reader) {
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: '', 
          timestamp: new Date(),
          toolCalls: []
        }]);
        
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data === '[DONE]') continue;
              
              try {
                const parsed = JSON.parse(data);
                if (parsed.choices?.[0]?.delta?.content) {
                  const content = parsed.choices[0].delta.content;
                  
                  // Debug logging
                  console.log('Received content:', JSON.stringify(content));
                  
                  // Check if this contains a tool call
                  if (content.includes('[Calling tool:')) {
                    console.log('Found tool call in content');
                    hasSeenToolCalls = true;
                    
                    const toolCallMatch = content.match(/\[Calling tool: ([^\]]+)\]/);
                    if (toolCallMatch) {
                      const toolCallText = toolCallMatch[1];
                      const [toolName, ...descParts] = toolCallText.split(' - ');
                      const toolDescription = descParts.join(' - ') || 'Running...';
                      
                      console.log('Extracted tool call:', { toolName, toolDescription });
                      
                      setMessages(prev => {
                        const newMessages = [...prev];
                        const lastMessage = newMessages[newMessages.length - 1];
                        if (lastMessage && lastMessage.role === 'assistant') {
                          if (!lastMessage.toolCalls) {
                            lastMessage.toolCalls = [];
                          }
                          
                          // Check for duplicates before adding
                          const toolCallText = `${toolName.trim()} - ${toolDescription.trim()}`;
                          const isDuplicate = lastMessage.toolCalls.some(
                            call => `${call.name} - ${call.description}` === toolCallText
                          );
                          
                          if (!isDuplicate) {
                            lastMessage.toolCalls.push({
                              name: toolName.trim(),
                              description: toolDescription.trim(),
                              timestamp: new Date()
                            });
                          }
                        }
                        return newMessages;
                      });
                    }
                  } else {
                    // Regular text content
                    console.log('Adding regular content:', JSON.stringify(content));
                    
                    // If we've seen tool calls and this is the first text after them,
                    // create a new assistant message
                    if (hasSeenToolCalls && content.trim()) {
                      console.log('Creating new message for post-tool content');
                      // Create a new assistant message for post-tool-call content
                      setMessages(prev => [...prev, { 
                        role: 'assistant', 
                        content: content,
                        timestamp: new Date(),
                        toolCalls: []
                      }]);
                      assistantMessage = content; // Reset for new message
                      hasSeenToolCalls = false; // Reset flag so subsequent text goes to this message
                    } else {
                      // Normal case - just add to current message
                      assistantMessage += content;
                      setMessages(prev => {
                        const newMessages = [...prev];
                        const lastMessage = newMessages[newMessages.length - 1];
                        if (lastMessage && lastMessage.role === 'assistant') {
                          lastMessage.content = assistantMessage;
                        }
                        return newMessages;
                      });
                    }
                  }
                }
              } catch (e) {
                console.error('Error parsing SSE data:', e);
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { 
        role: 'system', 
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error occurred'}`,
        timestamp: new Date()
      }]);
    } finally {
      setIsStreaming(false);
    }
  };

  const handleClear = () => {
    setMessages([]);
    setSettings(prev => ({ ...prev, researchIdea: '' }));
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <div className="w-96 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-6 border-b border-gray-200">
          <h1 className="text-2xl font-bold text-gray-800">Hypothesis Generator</h1>
          <p className="text-sm text-gray-600 mt-1">Generate research hypotheses with AI</p>
        </div>
        
        <form onSubmit={handleSubmit} className="flex-1 p-6 space-y-6 overflow-y-auto">
          {/* Domain Input */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Research Domain
            </label>
            <input
              type="text"
              value={settings.domain}
              onChange={(e) => setSettings(prev => ({ ...prev, domain: e.target.value }))}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white text-gray-900"
              placeholder="e.g., AI for Drug Discovery"
              disabled={isStreaming}
            />
            <p className="text-xs text-gray-500 mt-1">Specify your research field or area</p>
          </div>

          {/* Number of Hypotheses */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Number of Hypotheses
            </label>
            <div className="flex items-center space-x-3">
              <button
                type="button"
                onClick={() => setSettings(prev => ({ ...prev, numHypotheses: Math.max(1, prev.numHypotheses - 1) }))}
                disabled={isStreaming || settings.numHypotheses <= 1}
                className="w-10 h-10 rounded-lg border border-gray-300 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
              >
                <svg className="w-4 h-4 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
                </svg>
              </button>
              <input
                type="number"
                min="1"
                max="10"
                value={settings.numHypotheses}
                onChange={(e) => {
                  const val = parseInt(e.target.value);
                  if (!isNaN(val) && val >= 1 && val <= 10) {
                    setSettings(prev => ({ ...prev, numHypotheses: val }));
                  }
                }}
                className="w-20 px-3 py-2 text-center border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white text-gray-900 font-semibold"
                disabled={isStreaming}
              />
              <button
                type="button"
                onClick={() => setSettings(prev => ({ ...prev, numHypotheses: Math.min(10, prev.numHypotheses + 1) }))}
                disabled={isStreaming || settings.numHypotheses >= 10}
                className="w-10 h-10 rounded-lg border border-gray-300 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
              >
                <svg className="w-4 h-4 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
              </button>
            </div>
            <p className="text-xs text-gray-500 mt-1">How many hypotheses to generate (1-10)</p>
          </div>

          {/* Research Idea */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Research Idea
            </label>
            <textarea
              value={settings.researchIdea}
              onChange={(e) => setSettings(prev => ({ ...prev, researchIdea: e.target.value }))}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none bg-white text-gray-900"
              placeholder="Describe your research idea in detail..."
              rows={8}
              disabled={isStreaming}
            />
            <p className="text-xs text-gray-500 mt-1">
              {settings.researchIdea.length} characters
            </p>
          </div>

          {/* Action Buttons */}
          <div className="space-y-3">
            <button
              type="submit"
              disabled={isStreaming || !settings.researchIdea.trim()}
              className="w-full py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium transition-colors"
            >
              {isStreaming ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Generating Hypotheses...
                </span>
              ) : (
                'Generate Hypotheses'
              )}
            </button>
            
            {messages.length > 0 && !isStreaming && (
              <button
                type="button"
                onClick={handleClear}
                className="w-full py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 font-medium transition-colors"
              >
                Clear & Start New
              </button>
            )}
          </div>
        </form>

        {/* Footer */}
        <div className="p-4 border-t border-gray-200 text-xs text-gray-500 text-center">
          Powered by OpenAI Agents SDK
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col">
        {/* Header Bar */}
        <div className="bg-white border-b border-gray-200 px-6 py-4">
          <h2 className="text-lg font-semibold text-gray-800">Generated Hypotheses</h2>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center max-w-md">
                <svg className="mx-auto h-12 w-12 text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                <h3 className="text-lg font-medium text-gray-900 mb-1">No hypotheses generated yet</h3>
                <p className="text-gray-500">Fill in the form on the left and click "Generate Hypotheses" to get started.</p>
              </div>
            </div>
          ) : (
            <div className="space-y-6 max-w-4xl">
              {messages.map((message, index) => (
                <div key={index} className="animate-fadeIn">
                  {message.role === 'user' ? (
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                      <div className="flex items-center mb-2">
                        <svg className="h-5 w-5 text-blue-600 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                        </svg>
                        <span className="font-semibold text-blue-900">Your Input</span>
                        <span className="ml-auto text-xs text-blue-600">
                          {message.timestamp.toLocaleTimeString()}
                        </span>
                      </div>
                      <pre className="whitespace-pre-wrap font-sans text-sm text-gray-700">{message.content}</pre>
                    </div>
                  ) : message.role === 'assistant' ? (
                    <div className="space-y-3">
                      {/* Assistant Response */}
                      <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
                        <div className="flex items-center mb-3">
                          <svg className="h-5 w-5 text-green-600 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          <span className="font-semibold text-gray-900">Generated Hypotheses</span>
                          <span className="ml-auto text-xs text-gray-500">
                            {message.timestamp.toLocaleTimeString()}
                          </span>
                        </div>
                        <div className="prose prose-sm max-w-none">
                          {message.content ? (
                            <pre className="whitespace-pre-wrap font-sans text-gray-700">{message.content}</pre>
                          ) : (
                            <div className="text-gray-500 italic">
                              {message.toolCalls && message.toolCalls.length > 0 ? 
                                'Processing research...' : 
                                'Generating response...'}
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Tool Calls */}
                      {message.toolCalls && message.toolCalls.length > 0 && (
                        <div className="space-y-2">
                          {message.toolCalls.map((toolCall, toolIndex) => (
                            <div key={toolIndex} className="bg-amber-50 border border-amber-200 rounded-lg p-3 flex items-center">
                              <div className="flex items-center flex-1">
                                <svg className="h-4 w-4 text-amber-600 mr-2 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                </svg>
                                <div className="flex-1 min-w-0">
                                  <div className="text-sm font-medium text-amber-800">
                                    {toolCall.name.replace('Calling tool: ', '')}
                                  </div>
                                  <div className="text-xs text-amber-700 truncate">
                                    {toolCall.description}
                                  </div>
                                </div>
                                <div className="text-xs text-amber-600 ml-2">
                                  {toolCall.timestamp.toLocaleTimeString()}
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                      <div className="flex items-center mb-2">
                        <svg className="h-5 w-5 text-red-600 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span className="font-semibold text-red-900">Error</span>
                      </div>
                      <pre className="whitespace-pre-wrap font-sans text-sm text-red-700">{message.content}</pre>
                    </div>
                  )}
                </div>
              ))}
              
              {isStreaming && messages[messages.length - 1]?.role === 'assistant' && (
                <div className="flex items-center text-gray-500 text-sm">
                  <svg className="animate-spin h-4 w-4 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Generating...
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}