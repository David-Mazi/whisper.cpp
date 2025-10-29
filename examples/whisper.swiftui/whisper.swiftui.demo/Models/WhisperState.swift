import Foundation
import SwiftUI
import AVFoundation

@MainActor
class WhisperState: NSObject, ObservableObject, AVAudioRecorderDelegate {
    @Published var isModelLoaded = false
    @Published var messageLog = ""
    @Published var canTranscribe = false
    @Published var isRecording = false
    @Published var isRealTimeTranscribing = false
    
    private var whisperContext: WhisperContext?
    private let recorder = Recorder()
    private var recordedFile: URL? = nil
    private var audioPlayer: AVAudioPlayer?
    private var audioCapture: AudioCapture?
    
    // For chunked transcription
    private var chunks: [String] = []
    private var isTranscribingChunk = false
    private var currentTranscriptionTask: Task<Void, Never>?
    
    // Merge Strategy Constants
    private var rollingBuffer: [Float] = []           // 10-20s audio samples
    private var superBuffer: [Float] = []             // Accumulated stable audio
    private var superBufferText: String = ""          // Stable transcription
    private var rollingBufferText: String = ""        // Current rolling transcription

    // Timing constants
    private let rollingMin: Int = 160_000   // 10s at 16kHz
    private let rollingMax: Int = 320_000   // 20s at 16kHz
    private let stableLength: Int = 240_000 // 15s at 16kHz
    private let cutLength: Int = 160_000    // 10s Mark at 16kHz
    private let overlapLength: Int = 80_000 // 5s at 16kHz
    
    private var builtInModelUrl: URL? {
        Bundle.main.url(forResource: "ggml-base.en", withExtension: "bin", subdirectory: "models")
    }
    
    private var sampleUrl: URL? {
        Bundle.main.url(forResource: "david2", withExtension: "wav", subdirectory: "samples")
    }
    
    private enum LoadError: Error {
        case couldNotLocateModel
    }
    
    override init() {
        super.init()
        loadModel()
    }
    
    func loadModel(path: URL? = nil, log: Bool = true) {
        do {
            whisperContext = nil
            if (log) { messageLog += "Loading model...\n" }
            let modelUrl = path ?? builtInModelUrl
            if let modelUrl {
                whisperContext = try WhisperContext.createContext(path: modelUrl.path())
                if (log) { messageLog += "Loaded model \(modelUrl.lastPathComponent)\n" }
            } else {
                if (log) { messageLog += "Could not locate model\n" }
            }
            canTranscribe = true
        } catch {
            print(error.localizedDescription)
            if (log) { messageLog += "\(error.localizedDescription)\n" }
        }
    }

    func benchCurrentModel() async {
        if whisperContext == nil {
            messageLog += "Cannot bench without loaded model\n"
            return
        }
        messageLog += "Running benchmark for loaded model\n"
        let result = await whisperContext?.benchFull(modelName: "<current>", nThreads: Int32(min(4, cpuCount())))
        if (result != nil) { messageLog += result! + "\n" }
    }

    func bench(models: [Model]) async {
        let nThreads = Int32(min(4, cpuCount()))

//        messageLog += "Running memcpy benchmark\n"
//        messageLog += await WhisperContext.benchMemcpy(nThreads: nThreads) + "\n"
//
//        messageLog += "Running ggml_mul_mat benchmark with \(nThreads) threads\n"
//        messageLog += await WhisperContext.benchGgmlMulMat(nThreads: nThreads) + "\n"

        messageLog += "Running benchmark for all downloaded models\n"
        messageLog += "| CPU | OS | Config | Model | Th | FA | Enc. | Dec. | Bch5 | PP | Commit |\n"
        messageLog += "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        for model in models {
            loadModel(path: model.fileURL, log: false)
            if whisperContext == nil {
                messageLog += "Cannot bench without loaded model\n"
                break
            }
            let result = await whisperContext?.benchFull(modelName: model.name, nThreads: nThreads)
            if (result != nil) { messageLog += result! + "\n" }
        }
        messageLog += "Benchmarking completed\n"
    }
    
    func transcribeSample() async {
        if let sampleUrl {
            await transcribeAudio(sampleUrl)
        } else {
            messageLog += "Could not locate sample\n"
        }
    }
    
    private func transcribeAudio(_ url: URL) async {
        if (!canTranscribe) {
            return
        }
        guard let whisperContext else {
            return
        }
        
        do {
            canTranscribe = false
            messageLog += "Reading wave samples...\n"
            let data = try readAudioSamples(url)
            messageLog += "Transcribing data...\n"
            await whisperContext.fullTranscribe(samples: data)
            let text = await whisperContext.getTranscription()
            messageLog += "Done: \(text)\n"
        } catch {
            print(error.localizedDescription)
            messageLog += "\(error.localizedDescription)\n"
        }
        
        canTranscribe = true
    }
    
    private func readAudioSamples(_ url: URL) throws -> [Float] {
        stopPlayback()
        try startPlayback(url)
        return try decodeWaveFile(url)
    }
    
    func toggleRecord() async {
        if isRecording {
            await recorder.stopRecording()
            isRecording = false
            if let recordedFile {
                await transcribeAudio(recordedFile)
            }
        } else {
            requestRecordPermission { granted in
                if granted {
                    Task {
                        do {
                            self.stopPlayback()
                            let file = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
                                .appending(path: "output.wav")
                            try await self.recorder.startRecording(toOutputFile: file, delegate: self)
                            self.isRecording = true
                            self.recordedFile = file
                        } catch {
                            print(error.localizedDescription)
                            self.messageLog += "\(error.localizedDescription)\n"
                            self.isRecording = false
                        }
                    }
                }
            }
        }
    }
        
        // ... existing code ...
        
        func toggleRealtimeTranscription() async {
            if isRealTimeTranscribing {
                await stopRTTranscribing()
            } else {
                await startRTTranscribing()
            }
        }
        
        private func startRTTranscribing() async {
            requestRecordPermission { granted in
                if granted {
                    Task {
                        do {
                            // Create audio capture
                            self.audioCapture = AudioCapture(
                                chunkSizeSeconds: 1,
                                realtime: true
                            )
                            
                            // Set up chunk callback for real-time
                            await self.audioCapture?.setChunkCallback { [weak self] chunk in
                                await self?.transcribeChunk(chunk)
                            }
                            
                            try await self.audioCapture?.startCapture()
                            self.isRealTimeTranscribing = true
                            self.chunks = []
                            self.messageLog = "Transcription started...\n"
                            self.superBufferText = ""
                            
                        } catch {
                            self.messageLog = "Error starting transcription: \(error.localizedDescription)\n"
                        }
                    }
                }
            }
        }
        
        private func stopRTTranscribing() async {
            guard let audioCapture = audioCapture else { return }
            
            let finalSamples = await audioCapture.stopCapture()
            isRealTimeTranscribing = false
            
            // Wait for any in-progress chunk
            await currentTranscriptionTask?.value
            
            // Process any remaining audio
            if !finalSamples.isEmpty {
                await transcribeChunk(finalSamples)
            }
            
            // Merge all chunks using your convergence strategy
//            let finalText = mergeChunks(chunks)
//            let finalText = chunks.joined(separator: " ")
//            messageLog += "\n=== Final Transcription ===\n\(finalText)\n"
            
            self.audioCapture = nil
        }
        
        private func transcribeChunk(_ samples: [Float]) async {
            guard !isTranscribingChunk else {
                print("Already transcribing, skipping chunk")
                return
            }
                        
            guard let whisperContext = whisperContext else { return }
            
            isTranscribingChunk = true
            
            currentTranscriptionTask = Task {
                // Transcribe this chunk
                await whisperContext.fullTranscribe(samples: samples)
                let text = await whisperContext.getTranscription()
                
                chunks.append(text)
                
                // Phase 1: Before super buffer exists (buffer < 20s)
                if samples.count < rollingMax {
                    // UI shows: superBufferText + rollingBufferText only via merge
                    messageLog = "Transcription: \(mergeWithOverlap(oldText: superBufferText, newText: text, overlapSeconds: 10))\n"
                }
                
                // Phase 2: Cutting (buffer hits 20s)
                else {
                    // We trim buffer to keep calculations short and show superBufferText + rollingBufferText (merged)
                    superBufferText = mergeWithOverlap(oldText: superBufferText, newText: text, overlapSeconds: 10)
                    await self.audioCapture?.cutAudioBuffer(keepFrom: cutLength)
                    
                    messageLog = "Transcription: \(superBufferText)\n"
                }
            }
                
            await currentTranscriptionTask?.value
            isTranscribingChunk = false
        }
    
//    private func performCut() async {
//        // Extract first 15s as stable
//        let stableAudio = Array(audioBuffer.prefix(stableLength))
//        
//        // Transcribe stable portion
//        await whisperContext?.fullTranscribe(samples: stableAudio)
//        superBufferText = await whisperContext?.getTranscription() ?? ""
//        
//        // Keep samples for super buffer
//        superBuffer = stableAudio
//        
//        // Rolling buffer = last 10s (includes 5s overlap)
//        rollingBuffer = Array(audioBuffer.suffix(cutLength))
//        
//        // Clear main buffer
//        audioBuffer = rollingBuffer
//    }
    
    private func mergeWithOverlap(oldText: String, newText: String, overlapSeconds: Int) -> String {
        if oldText.isEmpty || newText.isEmpty {
            return oldText+newText
        }
        print("Old Text: \(oldText)\n")
        print("New Text: \(newText)\n")
        let oldWords = oldText.split(separator: " ").map(String.init)
        let newWords = newText.split(separator: " ").map(String.init)
        
        // Estimate words in overlap region (roughly 2-3 words per second)
        let overlapWordEstimate = overlapSeconds * 3
        
        
        // Find best matching substring
        if let (oldIdx, newIdx, matchLen) = findBestMatch(oldWords, newWords, overlapWordEstimate) {
            // Merge at middle of match
            let mergePoint = (matchLen+1) / 2
//            let mergePoint = matchLen / 2

            print("Match Len: \(matchLen)\n")
            
//            let keepFromOld = oldWords.count - overlapWordEstimate + oldIdx + mergePoint
            let keepFromOld = oldIdx + mergePoint

            let skipFromNew = newIdx + mergePoint
            print("Old")
            print(Array(oldWords.prefix(upTo: mergePoint+2)))
            print("New")
            print(Array(newWords.suffix(from: mergePoint-2)))
            let merged = Array(oldWords.prefix(keepFromOld)) +
                         Array(newWords.suffix(from: skipFromNew))
            print("Merged: \(merged)\n")
            return merged.joined(separator: " ")
        }
        
        // No good overlap found - simple concat with space
        return oldText + " " + newText
    }
    
    private func normalizeWord(_ word: String) -> String {
        return word.lowercased()
            .trimmingCharacters(in: .punctuationCharacters)
    }

    private func findBestMatch(_ oldWords: [String], _ newWords: [String], _ overlapWordEstimate: Int) -> (Int, Int, Int)? {
        var bestMatch: (oldIdx: Int, newIdx: Int, length: Int)? = nil
        var maxLen = 0
        
        // Define search regions
        let oldStartIdx = max(0, oldWords.count - overlapWordEstimate)
        let newEndIdx = min(newWords.count, overlapWordEstimate)
        
        // Create normalized versions for matching
        let oldWordsNormalized = oldWords.map { normalizeWord($0) }
        let newWordsNormalized = newWords.map { normalizeWord($0) }
        
        // Sliding window to find longest common substring
        for i in oldStartIdx..<oldWords.count {
            for j in 0..<newEndIdx {
                var len = 0
                while i + len < oldWords.count &&
                      j + len < newWords.count &&
                        oldWordsNormalized[i + len] == newWordsNormalized[j + len] {
                    len += 1
                }
                if len > maxLen && len >= 3 {  // Require at least 3 word match
                    maxLen = len
                    bestMatch = (i, j, len)
                }
            }
        }
        
        return bestMatch
    }
    
    private func findBestMatchWConfidence(_ arr1: [String], _ arr2: [String]) -> (Int, Int, Int)? {
        var bestMatch: (oldIdx: Int, newIdx: Int, length: Int)? = nil
        var maxLen = 0
        var bestConfidence = 0.0
        
        let minOverlapSize = min(arr1.count, arr2.count)
        
        for i in 0..<arr1.count {
            for j in 0..<arr2.count {
                var len = 0
                while i + len < arr1.count &&
                      j + len < arr2.count &&
                      arr1[i + len] == arr2[j + len] {
                    len += 1
                }
                
                if len >= 3 {
                    // Calculate confidence metrics
                    let lengthRatio = Double(len) / Double(minOverlapSize)  // 0.0 to 1.0
                    
                    // Prefer matches closer to center of overlap regions
                    let centerScore1 = 1.0 - abs(Double(i + len/2) - Double(arr1.count/2)) / Double(arr1.count)
                    let centerScore2 = 1.0 - abs(Double(j + len/2) - Double(arr2.count/2)) / Double(arr2.count)
                    let positionScore = (centerScore1 + centerScore2) / 2.0
                    
                    // Combined confidence (weighted toward length)
                    let confidence = (lengthRatio * 0.7) + (positionScore * 0.3)
                    
                    // Keep best match by length, use confidence as tiebreaker
                    if len > maxLen || (len == maxLen && confidence > bestConfidence) {
                        maxLen = len
                        bestConfidence = confidence
                        bestMatch = (i, j, len)
                    }
                }
            }
        }
        
        return bestMatch
    }
        
//        private func transcribeAudio(_ samples: [Float]) async {
//            guard let whisperContext = whisperContext else { return }
//            
//            canTranscribe = false
//            
//            await whisperContext.fullTranscribe(samples: samples)
//            let text = await whisperContext.getTranscription()
//            
//            messageLog += "Done: \(text)\n"
//            canTranscribe = true
//        }
        
        // Implement your merge strategy here
//        private func mergeChunks(_ chunks: [String]) -> String {
//            guard chunks.count > 1 else {
//                return chunks.first ?? ""
//            }
//            
//            var merged = chunks[0]
//            
//            for i in 1..<chunks.count {
//                merged = mergeWithConvergence(merged, chunks[i])
//            }
//            
//            return merged
//            return ""
//        }
        
//        private func mergeWithConvergence(_ seg1: String, _ seg2: String) -> String {
//            // Implement your convergence-based merge strategy here
//            // For now, simple concatenation
//            // TODO: Add your sliding window comparison logic
//            
//            let words1 = seg1.split(separator: " ").map(String.init)
//            let words2 = seg2.split(separator: " ").map(String.init)
//            
//            // Simple overlap detection (you'll enhance this)
//            let overlapSize = min(20, words1.count / 3)
//            let seg1End = Array(words1.suffix(overlapSize))
//            let seg2Start = Array(words2.prefix(overlapSize))
//            
//            // Find best match (simplified - use your algorithm)
//            if let overlapIndex = findOverlap(seg1End, seg2Start) {
//                let mergePoint1 = words1.count - overlapSize + overlapIndex
//                let mergePoint2 = overlapIndex
//                
//                let merged = words1[0..<mergePoint1] + words2[mergePoint2...]
//                return merged.joined(separator: " ")
//            }
//            
//            // No good overlap found, just concatenate
//            return seg1 + " " + seg2
//            return ""
//        }
        
//        private func findOverlap(_ arr1: [String], _ arr2: [String]) -> Int? {
//            // Simplified - implement your sliding window similarity here
//            for i in 0..<min(arr1.count, arr2.count) {
//                if arr1[i] == arr2[i] {
//                    return i
//                }
//            }
//            return nil
//            return 0
//        }
    
    private func requestRecordPermission(response: @escaping (Bool) -> Void) {
#if os(macOS)
        response(true)
#else
        AVAudioSession.sharedInstance().requestRecordPermission { granted in
            response(granted)
        }
#endif
    }
    
    private func startPlayback(_ url: URL) throws {
        audioPlayer = try AVAudioPlayer(contentsOf: url)
        audioPlayer?.play()
    }
    
    private func stopPlayback() {
        audioPlayer?.stop()
        audioPlayer = nil
    }
    
    // MARK: AVAudioRecorderDelegate
    
    nonisolated func audioRecorderEncodeErrorDidOccur(_ recorder: AVAudioRecorder, error: Error?) {
        if let error {
            Task {
                await handleRecError(error)
            }
        }
    }
    
    private func handleRecError(_ error: Error) {
        print(error.localizedDescription)
        messageLog += "\(error.localizedDescription)\n"
        isRecording = false
    }
    
    nonisolated func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        Task {
            await onDidFinishRecording()
        }
    }
    
    private func onDidFinishRecording() {
        isRecording = false
    }
}


fileprivate func cpuCount() -> Int {
    ProcessInfo.processInfo.processorCount
}
