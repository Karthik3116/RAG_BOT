import { BookOpen, ChevronLeft, ChevronRight } from 'lucide-react';

export default function ReferencePanel({
  sources,
  pageIndex,
  onPageIndexChange,
  selectedMessageIndex
}) {
  const handleNext = () => {
    if (pageIndex < sources.length - 1) {
      onPageIndexChange(pageIndex + 1);
    }
  };

  const handlePrev = () => {
    if (pageIndex > 0) {
      onPageIndexChange(pageIndex - 1);
    }
  };

  const currentSource = sources?.[pageIndex];

  return (
    <div className="w-1/3 bg-white border-l border-slate-200 p-6 flex flex-col h-full">
      <h2 className="text-xl font-bold mb-4 flex items-center text-slate-700"><BookOpen size={20} className="mr-2" /> References</h2>
      <div className="flex-1 bg-slate-50 rounded-lg p-4 overflow-y-auto flex flex-col">
        {sources && sources.length > 0 ? (
          <div className="flex flex-col h-full">
            <div className="text-center mb-2 font-semibold text-slate-600">
              {`Source ${pageIndex + 1} of ${sources.length}`}
            </div>
            <div className="flex-1 mb-4 overflow-hidden flex flex-col">
              <h3 className="font-bold text-sm truncate mb-2">{currentSource.source}, Page {currentSource.page}</h3>
              <div className="flex-1 flex items-center justify-center overflow-hidden">
                <img src={currentSource.base64Image} alt={`Source page ${currentSource.page}`} className="w-full h-full object-contain rounded-md border border-slate-200" />
              </div>
            </div>
            <div className="flex justify-between items-center mt-auto pt-2 border-t border-slate-200">
              <button onClick={handlePrev} disabled={pageIndex === 0} className="px-3 py-1 bg-slate-200 rounded-md hover:bg-slate-300 disabled:opacity-50 flex items-center transition-colors"><ChevronLeft size={16} className="mr-1" /> Previous</button>
              <button onClick={handleNext} disabled={pageIndex === sources.length - 1} className="px-3 py-1 bg-slate-200 rounded-md hover:bg-slate-300 disabled:opacity-50 flex items-center transition-colors">Next <ChevronRight size={16} className="ml-1" /></button>
            </div>
          </div>
        ) : (
          <div className="text-center text-slate-500 h-full flex items-center justify-center">
            <p>{selectedMessageIndex !== null ? "Failed to highlight the information." : "Click 'Show Sources' on a message to see its references here."}</p>
          </div>
        )}
      </div>
    </div>
  );
}