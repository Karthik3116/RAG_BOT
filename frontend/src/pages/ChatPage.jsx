import { useState, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import axios from 'axios';
import Sidebar from '../components/Sidebar';
import ChatView from '../components/ChatView';
import ReferencePanel from '../components/ReferencePanel';
import { useSessionStorage } from '../hooks/useSessionStorage';

const API_URL = "http://127.0.0.1:5001";

const createNewChat = () => ({
  title: 'New Chat',
  messages: [],
  processingComplete: false,
  processedFileNames: [], // Changed from `files` to be more descriptive
});

export default function ChatPage({ apiKey }) {
  const [chats, setChats] = useSessionStorage('chatSessions', {});
  const [activeChatId, setActiveChatId] = useSessionStorage('activeChatId', null);
  const [files, setFiles] = useState([]); // This now only tracks newly selected files
  const [processing, setProcessing] = useState(false);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedMessageIndex, setSelectedMessageIndex] = useState(null);
  const [sourcePageIndex, setSourcePageIndex] = useState(0);

  // Initialize first chat if none exist
  useEffect(() => {
    if (!activeChatId || !chats[activeChatId]) {
      const newId = uuidv4();
      const newChat = createNewChat();
      setChats({ [newId]: newChat });
      setActiveChatId(newId);
    }
  }, [chats, activeChatId, setChats, setActiveChatId]);

  const activeChat = chats[activeChatId] || createNewChat();

  const handleNewChat = () => {
    const newId = uuidv4();
    setChats(prev => ({ ...prev, [newId]: createNewChat() }));
    setActiveChatId(newId);
    setFiles([]);
    setSelectedMessageIndex(null);
  };

  const handleSelectChat = (chatId) => {
    setActiveChatId(chatId);
    setFiles([]); // Reset file input when switching chats
    setSelectedMessageIndex(null);
  };
  
  const handleDeleteChat = (chatIdToDelete) => {
    const newChats = { ...chats };
    delete newChats[chatIdToDelete];
    setChats(newChats);

    if (activeChatId === chatIdToDelete) {
      const remainingIds = Object.keys(newChats);
      if (remainingIds.length > 0) {
        setActiveChatId(remainingIds[0]);
      } else {
        handleNewChat();
      }
    }
  };

  const handleFileChange = (event) => {
    setFiles([...event.target.files]);
  };

  const handleProcess = async () => {
    if (files.length === 0) {
      alert("Please upload at least one PDF for this chat.");
      return;
    }
    setProcessing(true);
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    formData.append('apiKey', apiKey);

    try {
      await axios.post(`${API_URL}/api/process-pdfs`, formData);
      setChats(prev => ({
        ...prev,
        [activeChatId]: { 
          ...activeChat, 
          processingComplete: true, 
          processedFileNames: files.map(f => f.name) // Store the processed file names
        }
      }));
      setFiles([]); // Clear the selected files after processing
    } catch (error) {
      console.error("Error processing documents:", error);
      alert(`Failed to process documents: ${error.response?.data?.error || error.message}`);
    } finally {
      setProcessing(false);
    }
  };

  const handleAsk = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    // === FIX START: Extract the last question from the input ===
    // This handles cases where users paste text or use shift+enter, ensuring
    // we only process the most recent question.
    const possibleQuestions = query.trim().split('\n').filter(line => line.trim().length > 0);
    const actualQuestion = possibleQuestions.length > 0 ? possibleQuestions[possibleQuestions.length - 1] : '';

    if (!actualQuestion) {
        setQuery(''); // Clear the input if it only contained whitespace
        return;
    }
    // === FIX END ===

    const userMessage = { role: 'user', content: actualQuestion }; // Use the cleaned question
    const updatedMessages = [...activeChat.messages, userMessage];
    
    // Optimistically update the chat with the user's message
    setChats(prev => ({ ...prev, [activeChatId]: { ...activeChat, messages: updatedMessages } }));
    
    setLoading(true);
    setQuery('');
    setSelectedMessageIndex(null);

    try {
      // Send only the cleaned, actual question to the backend
      const response = await axios.post(`${API_URL}/api/ask`, { question: actualQuestion, apiKey });
      const { answer, sources } = response.data;
      const botMessage = { role: 'bot', content: answer, sources: sources || [] };
      
      setChats(prev => {
        const currentChat = prev[activeChatId];
        const newMessages = [...currentChat.messages, botMessage];
        // Update title with the first user message if it's a new chat
        const newTitle = currentChat.messages.length === 1 ? userMessage.content.substring(0, 30) + '...' : currentChat.title;
        return { ...prev, [activeChatId]: { ...currentChat, messages: newMessages, title: newTitle } };
      });
      setSelectedMessageIndex(updatedMessages.length);
      setSourcePageIndex(0);

    } catch (error) {
      console.error("Error asking question:", error);
      const errorMessage = { role: 'bot', content: "Sorry, I encountered an error. Please try again.", sources: [] };
      setChats(prev => ({ ...prev, [activeChatId]: { ...activeChat, messages: [...updatedMessages, errorMessage] } }));
    } finally {
      setLoading(false);
    }
  };

  const handleShowSources = (messageIndex) => {
    setSelectedMessageIndex(messageIndex);
    setSourcePageIndex(0);
  };

  const selectedSources = selectedMessageIndex !== null && activeChat.messages[selectedMessageIndex]?.sources;

  return (
    <div className="min-h-screen bg-slate-100 text-slate-800 flex font-sans">
      <Sidebar
        chats={chats}
        activeChatId={activeChatId}
        onNewChat={handleNewChat}
        onSelectChat={handleSelectChat}
        onDeleteChat={handleDeleteChat}
        files={files}
        onFileChange={handleFileChange}
        processing={processing}
        onProcess={handleProcess}
        processedFileNames={activeChat.processedFileNames} 
      />
      <main className="flex-1 flex h-screen">
        <ChatView
          messages={activeChat.messages}
          loading={loading}
          query={query}
          onQueryChange={(e) => setQuery(e.target.value)}
          onAsk={handleAsk}
          isProcessingComplete={activeChat.processingComplete}
          onShowSources={handleShowSources}
        />
        <ReferencePanel
          sources={selectedSources}
          pageIndex={sourcePageIndex}
          onPageIndexChange={setSourcePageIndex}
          selectedMessageIndex={selectedMessageIndex}
        />
      </main>
    </div>
  );
}