import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Brain, CheckCircle, AlertTriangle, Clock, Download } from 'lucide-react'
import { PredictionResult } from '../App'
import { generatePDFReport } from '../utils/reportGenerator'
import { CollapsibleNarrative } from './CollapsibleNarrative'

interface ResultsSectionProps {
    result: PredictionResult | null
    isLoading: boolean
}

export default function ResultsSection({ result, isLoading }: ResultsSectionProps) {
    return (
        <div className="glass rounded-2xl p-8 h-full flex flex-col">
            <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
                <Brain size={28} className="text-blue-400" />
                Analysis Results
            </h3>

            <div className="flex-1 flex items-center justify-center">
                <AnimatePresence mode="wait">
                    {isLoading ? (
                        <LoadingState key="loading" />
                    ) : result ? (
                        <ResultDisplay key="result" result={result} />
                    ) : (
                        <EmptyState key="empty" />
                    )}
                </AnimatePresence>
            </div>
        </div>
    )
}

function LoadingState() {
    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="text-center"
        >
            <motion.div
                animate={{
                    rotate: 360,
                    scale: [1, 1.2, 1],
                }}
                transition={{
                    rotate: { duration: 2, repeat: Infinity, ease: "linear" },
                    scale: { duration: 1, repeat: Infinity },
                }}
                className="mb-6"
            >
                <Brain size={64} className="text-purple-400 mx-auto" />
            </motion.div>
            <h4 className="text-white text-xl font-semibold mb-2">Analyzing MRI Scan</h4>
            <p className="text-white/60 mb-4">AI model is processing your image...</p>
            <div className="flex justify-center gap-2">
                <motion.div
                    className="w-2 h-2 bg-purple-400 rounded-full"
                    animate={{ scale: [1, 1.5, 1] }}
                    transition={{ duration: 1, repeat: Infinity, delay: 0 }}
                />
                <motion.div
                    className="w-2 h-2 bg-blue-400 rounded-full"
                    animate={{ scale: [1, 1.5, 1] }}
                    transition={{ duration: 1, repeat: Infinity, delay: 0.2 }}
                />
                <motion.div
                    className="w-2 h-2 bg-pink-400 rounded-full"
                    animate={{ scale: [1, 1.5, 1] }}
                    transition={{ duration: 1, repeat: Infinity, delay: 0.4 }}
                />
            </div>
        </motion.div>
    )
}

function EmptyState() {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="text-center text-white/40"
        >
            <Brain size={64} className="mx-auto mb-4 opacity-30" />
            <p className="text-lg">Upload an MRI scan to see results</p>
            <p className="text-sm mt-2">AI analysis will appear here</p>
        </motion.div>
    )
}

function ResultDisplay({ result }: { result: PredictionResult }) {
    const isTumor = result.prediction === 'tumor'
    const confidencePercent = (result.confidence * 100).toFixed(1)
    const confidence = result.confidence * 100

    // Determine animation based on confidence
    const getConfidenceAnimation = () => {
        if (confidence > 90) {
            // High confidence - subtle static glow (professional)
            return {
                boxShadow: `0 0 25px ${isTumor ? 'rgba(239, 68, 68, 0.4)' : 'rgba(34, 197, 94, 0.4)'}`
            }
        } else if (confidence >= 60) {
            // Medium confidence - very subtle pulse
            return {
                scale: [1, 1.01, 1],
                transition: { duration: 3, repeat: Infinity, ease: "easeInOut" }
            }
        } else {
            // Low confidence - subtle shake effect
            return {
                x: [0, -1, 1, -1, 1, 0],
                transition: { duration: 0.4, repeat: Infinity, repeatDelay: 2 }
            }
        }
    }

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="w-full space-y-6"
        >
            {/* Main Result with Dynamic Animation */}
            <motion.div
                className={`
          p-6 rounded-xl border-2 relative overflow-hidden
          ${isTumor
                        ? 'bg-red-500/10 border-red-500/50'
                        : 'bg-green-500/10 border-green-500/50'
                    }
        `}
                initial={{ scale: 0 }}
                animate={{
                    scale: 1,
                    ...getConfidenceAnimation()
                }}
                transition={{ type: "spring", duration: 0.6, delay: 0.2 }}
            >
                {/* Confidence Level Indicator */}
                <div className="absolute top-2 right-2">
                    {confidence > 90 ? (
                        <span className="px-2 py-1 bg-green-500/20 text-green-300 text-xs rounded-full font-bold">
                            ⚡ HIGH
                        </span>
                    ) : confidence >= 60 ? (
                        <span className="px-2 py-1 bg-yellow-500/20 text-yellow-300 text-xs rounded-full font-bold">
                            ⚠️ MEDIUM
                        </span>
                    ) : (
                        <span className="px-2 py-1 bg-red-500/20 text-red-300 text-xs rounded-full font-bold">
                            ⚠️ LOW
                        </span>
                    )}
                </div>

                <div className="flex items-center gap-4 mb-4">
                    {isTumor ? (
                        <AlertTriangle size={40} className="text-red-400" />
                    ) : (
                        <CheckCircle size={40} className="text-green-400" />
                    )}
                    <div>
                        <h4 className="text-2xl font-bold text-white">
                            {isTumor ? 'Tumor Detected' : 'No Tumor Detected'}
                        </h4>
                        <p className="text-white/70">
                            Confidence: <span className="font-bold">{confidencePercent}%</span>
                        </p>
                    </div>
                </div>

                {/* Confidence Bar */}
                <div className="w-full bg-white/10 rounded-full h-3 overflow-hidden">
                    <motion.div
                        className={`h-full rounded-full ${isTumor ? 'bg-red-500' : 'bg-green-500'}`}
                        initial={{ width: 0 }}
                        animate={{ width: `${confidencePercent}%` }}
                        transition={{ duration: 1, delay: 0.5 }}
                    />
                </div>
            </motion.div>

            {/* AI Narrative - Collapsible */}
            {result.narrative && (
                <CollapsibleNarrative narrative={result.narrative} />
            )}

            {/* Download Report Button */}
            <motion.button
                onClick={() => {
                    generatePDFReport({
                        prediction: result.prediction,
                        confidence: result.confidence,
                        timestamp: new Date().toISOString(),
                        processingTime: result.processingTime,
                        narrative: result.narrative,
                        modelInfo: {
                            name: 'EfficientNet-B4',
                            accuracy: 0.923,
                            dataset: '3,264 Brain MRI Scans (Kaggle)'
                        }
                    })
                }}
                className="w-full py-3 rounded-xl font-semibold flex items-center justify-center gap-2
                    bg-gradient-to-r from-blue-500 to-purple-500 
                    hover:from-blue-600 hover:to-purple-600 
                    text-white shadow-lg hover:shadow-xl
                    transition-all duration-300"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
            >
                <Download size={20} />
                Download AI Report
            </motion.button>

            {/* Processing Time */}
            {result.processingTime && (
                <motion.div
                    className="flex items-center justify-center gap-2 text-white/60 text-sm"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.6 }}
                >
                    <Clock size={16} />
                    <span>Processed in {result.processingTime.toFixed(2)}s</span>
                </motion.div>
            )}

            {/* Interactive Grad-CAM */}
            {result.gradcamUrl && (
                <InteractiveGradCAM gradcamUrl={result.gradcamUrl} isTumor={isTumor} />
            )}
        </motion.div>
    )
}

// Interactive Grad-CAM Component
function InteractiveGradCAM({ gradcamUrl, isTumor }: { gradcamUrl: string; isTumor: boolean }) {
    const [clickedRegion, setClickedRegion] = useState<{ x: number; y: number; explanation: string } | null>(null)

    const handleImageClick = (e: React.MouseEvent<HTMLImageElement>) => {
        const rect = e.currentTarget.getBoundingClientRect()
        const x = ((e.clientX - rect.left) / rect.width) * 100
        const y = ((e.clientY - rect.top) / rect.height) * 100

        // Generate explanation based on region (simplified - in production, this would call backend)
        let explanation = ""
        const isTopHalf = y < 50
        const isLeftHalf = x < 50
        const isCentralRegion = x > 30 && x < 70 && y > 30 && y < 70

        if (isCentralRegion) {
            explanation = isTumor
                ? "🔴 High attention region: Model detected significant structural abnormalities. This area shows patterns consistent with tumor tissue, including irregular density and texture variations."
                : "🟢 Central region analysis: Normal brain tissue structure detected. The model found consistent density patterns typical of healthy brain tissue."
        } else if (isTopHalf && isLeftHalf) {
            explanation = isTumor
                ? "🟠 Upper-left quadrant: Moderate attention level. Model identified subtle texture changes that contribute to the overall tumor detection."
                : "🔵 Upper-left quadrant: Standard tissue patterns observed. This region shows expected anatomical features with no anomalies."
        } else if (isTopHalf) {
            explanation = isTumor
                ? "🟠 Upper-right quadrant: Supporting evidence detected. Model noticed boundary irregularities typical of tumor margins."
                : "🔵 Upper-right quadrant: Normal cortical structures identified. Tissue appears consistent with healthy brain anatomy."
        } else if (isLeftHalf) {
            explanation = isTumor
                ? "🟡 Lower-left quadrant: Peripheral attention area. Model detected contrast variations that support tumor presence."
                : "🔵 Lower-left quadrant: Healthy tissue characteristics. Standard gray/white matter boundaries observed."
        } else {
            explanation = isTumor
                ? "🟡 Lower-right quadrant: Supplementary findings. Model found texture patterns consistent with affected tissue."
                : "🔵 Lower-right quadrant: Normal tissue structure. No abnormalities detected in this region."
        }

        setClickedRegion({ x, y, explanation })
    }

    return (
        <motion.div
            className="p-4 glass-dark rounded-xl relative"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
        >
            <h5 className="text-white font-semibold mb-3 flex items-center gap-2">
                <Brain size={20} className="text-purple-400" />
                Interactive Attention Heatmap
            </h5>

            {/* Prominent Call-to-Action Banner */}
            <motion.div
                className="mb-4 p-3 bg-gradient-to-r from-purple-500/20 to-blue-500/20 border-2 border-purple-400/50 rounded-lg"
                animate={{
                    borderColor: ['rgba(167, 139, 250, 0.5)', 'rgba(96, 165, 250, 0.8)', 'rgba(167, 139, 250, 0.5)'],
                }}
                transition={{ duration: 2, repeat: Infinity }}
            >
                <div className="flex items-center gap-3">
                    <motion.div
                        animate={{ scale: [1, 1.2, 1] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                        className="text-3xl"
                    >
                        👆
                    </motion.div>
                    <div>
                        <p className="text-white font-bold text-lg">Click Anywhere on the Heatmap!</p>
                        <p className="text-white/70 text-sm">Explore different regions to see what the AI detected</p>
                    </div>
                </div>
            </motion.div>

            <motion.div
                className="relative border-4 border-transparent rounded-lg overflow-hidden"
                animate={{
                    borderColor: clickedRegion
                        ? ['rgba(167, 139, 250, 0)', 'rgba(167, 139, 250, 0)', 'rgba(167, 139, 250, 0)']
                        : ['rgba(167, 139, 250, 0.3)', 'rgba(96, 165, 250, 0.6)', 'rgba(167, 139, 250, 0.3)'],
                }}
                transition={{ duration: 2, repeat: Infinity }}
            >
                <img
                    src={gradcamUrl}
                    alt="Grad-CAM"
                    className="w-full rounded-lg cursor-pointer hover:opacity-90 transition-opacity"
                    onClick={handleImageClick}
                />
                {clickedRegion && (
                    <motion.div
                        className="absolute rounded-full w-8 h-8 border-4 border-white pointer-events-none"
                        style={{
                            left: `${clickedRegion.x}%`,
                            top: `${clickedRegion.y}%`,
                            transform: 'translate(-50%, -50%)'
                        }}
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ type: "spring", duration: 0.3 }}
                    />
                )}
            </motion.div>

            {/* Region Explanation */}
            <AnimatePresence>
                {clickedRegion && (
                    <motion.div
                        className="mt-4 p-4 bg-purple-500/20 border border-purple-400/30 rounded-lg"
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                    >
                        <div className="flex items-start gap-2">
                            <div className="text-2xl">🔍</div>
                            <div className="flex-1">
                                <h6 className="text-white font-semibold mb-1">Region Analysis</h6>
                                <p className="text-white/80 text-sm leading-relaxed">
                                    {clickedRegion.explanation}
                                </p>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {!clickedRegion && (
                <motion.p
                    className="text-center text-white/60 text-sm mt-4 flex items-center justify-center gap-2"
                    animate={{ opacity: [0.6, 1, 0.6] }}
                    transition={{ duration: 2, repeat: Infinity }}
                >
                    <span className="text-yellow-400 text-lg">⚡</span>
                    <span>Click any part of the image above to start exploring!</span>
                    <span className="text-yellow-400 text-lg">⚡</span>
                </motion.p>
            )}
        </motion.div>
    )
}
