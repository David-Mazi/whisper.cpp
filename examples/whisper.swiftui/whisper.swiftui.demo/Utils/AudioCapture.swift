//
//  AudioCapture.swift
//  whisper.swiftui
//
//  Created by David Mazi on 10/28/25.
//

import Foundation
import AVFoundation


actor AudioCapture {
    private var audioEngine: AVAudioEngine?
    private var audioBuffer: [Float] = []
    private var lastProcessedCount: Int = 0
//    private let bufferLock = NSLock()
    
    // Callback for when new chunk is ready
    private var onChunkReady: (([Float]) async -> Void)?

    // Real-time vs batch mode
    private var isRealtime: Bool = false
    private let chunkSize: Int
    private let sampleRate: Double = 16000
    
    
    init(chunkSizeSeconds: Int = 10, realtime: Bool = false) {
        self.chunkSize = Int(Double(chunkSizeSeconds) * sampleRate)
        self.isRealtime = realtime
    }
    
    // Add this method to set the callback
    func setChunkCallback(_ callback: @escaping ([Float]) async -> Void) {
        self.onChunkReady = callback
    }
    
    // Update Buffer
    func cutAudioBuffer(keepFrom: Int) async {
        if audioBuffer.count > keepFrom {
            self.audioBuffer = Array(audioBuffer.suffix(from: keepFrom))
        }
        self.lastProcessedCount = audioBuffer.count
    }
    
    func startCapture() throws {
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.record, mode: .measurement)
        try audioSession.setActive(true)
        
        audioEngine = AVAudioEngine()
        guard let audioEngine = audioEngine else { return }
        
        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)
        
        // Create 16kHz mono format (Whisper requirement)
        guard let recordingFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw AudioCaptureError.formatError
        }
        
        // Install tap to capture audio samples
        inputNode.installTap(
            onBus: 0,
            bufferSize: 4096,
            format: inputFormat
        ) { [weak self] buffer, time in
            guard let self = self else { return }
            
            // Convert to Whisper format (16kHz mono)
//            self.processAudioBuffer(buffer,
//                                   inputFormat: inputFormat,
//                                   targetFormat: recordingFormat)
            // Process on audio thread, then append via actor isolation
            let samples = self.extractSamples(from: buffer,
                                             inputFormat: inputFormat,
                                             targetFormat: recordingFormat)
                        
            // Move to actor context for thread-safe buffer access
            Task {
                await self.appendSamples(samples)
            }
        }
        
        audioEngine.prepare()
        try audioEngine.start()
    }
    
    // NOT actor-isolated - runs on audio thread
    private nonisolated func extractSamples(from buffer: AVAudioPCMBuffer,
                                           inputFormat: AVAudioFormat,
                                           targetFormat: AVAudioFormat) -> [Float] {
        let convertedBuffer = convertBuffer(buffer,
                                           from: inputFormat,
                                           to: targetFormat)
        
        guard let floatData = convertedBuffer.floatChannelData?[0] else {
            return []
        }
        let frameLength = Int(convertedBuffer.frameLength)
        return Array(UnsafeBufferPointer(start: floatData, count: frameLength))
    }
    
    // Actor-isolated - thread-safe
    private func appendSamples(_ samples: [Float]) async {
        audioBuffer.append(contentsOf: samples)
        
        // Only trigger callback when we have chunkSize NEW samples since last process
        let newSampleCount = audioBuffer.count - lastProcessedCount
        
        // For your first pass: process all available audio
        if isRealtime && newSampleCount >= chunkSize {
            await onChunkReady?(audioBuffer)
            lastProcessedCount = audioBuffer.count
        }
        
//        // Check if we have enough for a chunk
//        if isRealtime && audioBuffer.count >= chunkSize {
//            processChunk()
//        }
//        let chunk = Array(audioBuffer.prefix(chunkSize))
//        audioBuffer.removeFirst(min(chunkSize, audioBuffer.count))
    }
    
//    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer,
//                                   inputFormat: AVAudioFormat,
//                                   targetFormat: AVAudioFormat) {
//        // Convert to target format if needed
//        let convertedBuffer = convertBuffer(buffer,
//                                           from: inputFormat,
//                                           to: targetFormat)
//        
//        guard let floatData = convertedBuffer.floatChannelData?[0] else { return }
//        let frameLength = Int(convertedBuffer.frameLength)
//        let samples = Array(UnsafeBufferPointer(start: floatData, count: frameLength))
//        
//        bufferLock.lock()
//        audioBuffer.append(contentsOf: samples)
//        bufferLock.unlock()
//        
//        // Check if we have enough for a chunk
//        if isRealtime && audioBuffer.count >= chunkSize {
//            processChunk()
//        }
//    }
    
    private nonisolated func convertBuffer(_ buffer: AVAudioPCMBuffer,
                              from: AVAudioFormat,
                              to: AVAudioFormat) -> AVAudioPCMBuffer {
        // If already correct format, return as-is
        if from.sampleRate == to.sampleRate && from.channelCount == to.channelCount {
            return buffer
        }
        
        // Create converter
        guard let converter = AVAudioConverter(from: from, to: to),
              let convertedBuffer = AVAudioPCMBuffer(
                pcmFormat: to,
                frameCapacity: AVAudioFrameCount(Double(buffer.frameLength) *
                                                 to.sampleRate / from.sampleRate)
              ) else {
            print("⚠️ Audio conversion failed: \(from.sampleRate)Hz → \(to.sampleRate)Hz")
            return buffer
        }
        
        var error: NSError?
        converter.convert(to: convertedBuffer, error: &error) { _, outStatus in
            outStatus.pointee = .haveData
            return buffer
        }
                
        if let error = error {
            print("⚠️ Conversion error: \(error)")
        }
        
        return convertedBuffer
    }
    
    private func processChunk() {
//        bufferLock.lock()
//        let chunk = Array(audioBuffer.prefix(chunkSize))
//        audioBuffer.removeFirst(min(chunkSize, audioBuffer.count))
//        bufferLock.unlock()
        
        // Trigger callback with chunk
        Task {
            await onChunkReady?(audioBuffer)
//            await onChunkReady?(chunk)
        }
    }
    
    func stopCapture() -> [Float] {
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        
//        bufferLock.lock()
        let finalBuffer = audioBuffer
        audioBuffer = []
        lastProcessedCount = 0
//        bufferLock.unlock()
        
        return finalBuffer
    }
    
//    func getAllSamples() -> [Float] {
//        bufferLock.lock()
//        defer { bufferLock.unlock() }
//        return audioBuffer
//    }
}

enum AudioCaptureError: Error {
    case formatError
    case conversionError
}
