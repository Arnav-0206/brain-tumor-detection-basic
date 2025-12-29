/**
 * Generate a professional PDF report for AI analysis
 */

import { jsPDF } from 'jspdf'

export interface ReportData {
    prediction: 'tumor' | 'no_tumor'
    confidence: number
    timestamp: string
    processingTime?: number
    narrative?: string
    modelInfo: {
        name: string
        accuracy: number
        dataset: string
    }
}

export function generatePDFReport(data: ReportData) {
    const doc = new jsPDF()
    const confidencePercent = (data.confidence * 100).toFixed(1)
    const date = new Date().toLocaleString()
    const reportId = `NS-${Date.now()}`

    let yPos = 20
    const pageWidth = doc.internal.pageSize.width
    const pageHeight = doc.internal.pageSize.height
    const margin = 20
    const contentWidth = pageWidth - (2 * margin)

    // Helper function to add text with word wrap
    const addText = (text: string, x: number, y: number, options: any = {}) => {
        const lines = doc.splitTextToSize(text, options.maxWidth || contentWidth)
        doc.text(lines, x, y)
        return y + (lines.length * (options.lineHeight || 6))
    }

    // Helper to check page break
    const checkPageBreak = (requiredSpace: number) => {
        if (yPos + requiredSpace > pageHeight - 30) {
            doc.addPage()
            yPos = 20
            return true
        }
        return false
    }

    // ========== HEADER ==========
    doc.setFillColor(88, 28, 135) // Purple
    doc.rect(0, 0, pageWidth, 40, 'F')

    doc.setTextColor(255, 255, 255)
    doc.setFontSize(26)
    doc.setFont('helvetica', 'bold')
    doc.text('NEUROSCAN AI', pageWidth / 2, 18, { align: 'center' })

    doc.setFontSize(11)
    doc.setFont('helvetica', 'normal')
    doc.text('Brain Tumor Detection System - AI Analysis Report', pageWidth / 2, 30, { align: 'center' })

    yPos = 50

    // ========== REPORT METADATA ==========
    doc.setFontSize(8)
    doc.setTextColor(100, 100, 100)
    doc.setFont('helvetica', 'italic')
    doc.text(`Generated: ${date}`, margin, yPos)
    doc.text(`Report ID: ${reportId}`, pageWidth - margin, yPos, { align: 'right' })
    yPos += 8

    // Divider
    doc.setDrawColor(200, 200, 200)
    doc.setLineWidth(0.3)
    doc.line(margin, yPos, pageWidth - margin, yPos)
    yPos += 12

    // ========== ANALYSIS SUMMARY SECTION ==========
    // Section Header
    doc.setFillColor(139, 92, 246) // Purple
    doc.rect(margin, yPos, contentWidth, 8, 'F')
    doc.setTextColor(255, 255, 255)
    doc.setFontSize(12)
    doc.setFont('helvetica', 'bold')
    doc.text('ANALYSIS SUMMARY', margin + 3, yPos + 5.5)
    yPos += 12

    // Summary Box
    doc.setFillColor(248, 250, 252)
    doc.rect(margin, yPos, contentWidth, 28, 'F')
    doc.setDrawColor(200, 200, 220)
    doc.setLineWidth(0.5)
    doc.rect(margin, yPos, contentWidth, 28)

    // Prediction Result
    const resultColor: [number, number, number] = data.prediction === 'tumor' ? [220, 38, 38] : [34, 197, 94]
    doc.setTextColor(resultColor[0], resultColor[1], resultColor[2])
    doc.setFontSize(14)
    doc.setFont('helvetica', 'bold')
    doc.text(data.prediction === 'tumor' ? 'TUMOR DETECTED' : 'NO TUMOR DETECTED', margin + 5, yPos + 7)

    // Confidence & Time
    doc.setTextColor(60, 60, 60)
    doc.setFontSize(10)
    doc.setFont('helvetica', 'normal')
    doc.text(`Confidence: ${confidencePercent}% (${data.confidence > 0.9 ? 'HIGH' : data.confidence >= 0.6 ? 'MEDIUM' : 'LOW'})`, margin + 5, yPos + 15)
    if (data.processingTime) {
        doc.text(`Processing Time: ${data.processingTime.toFixed(2)}s`, margin + 5, yPos + 21)
    }
    yPos += 35

    // ========== MODEL INFORMATION SECTION ==========
    checkPageBreak(50)

    doc.setFillColor(59, 130, 246) // Blue
    doc.rect(margin, yPos, contentWidth, 8, 'F')
    doc.setTextColor(255, 255, 255)
    doc.setFontSize(12)
    doc.setFont('helvetica', 'bold')
    doc.text('AI MODEL INFORMATION', margin + 3, yPos + 5.5)
    yPos += 12

    doc.setFontSize(10)
    doc.setTextColor(40, 40, 40)
    doc.setFont('helvetica', 'normal')

    const modelData = [
        ['Architecture:', data.modelInfo.name],
        ['Training Accuracy:', `${(data.modelInfo.accuracy * 100).toFixed(1)}%`],
        ['Dataset:', data.modelInfo.dataset],
        ['Input Resolution:', '224x224 pixels'],
        ['Framework:', 'PyTorch + timm'],
        ['Explainability:', 'Grad-CAM visualization']
    ]

    modelData.forEach(([label, value]) => {
        doc.setFont('helvetica', 'bold')
        doc.text(label, margin + 3, yPos)
        doc.setFont('helvetica', 'normal')
        doc.text(value, margin + 45, yPos)
        yPos += 5
    })
    yPos += 8

    // ========== DATASET STATISTICS SECTION ==========
    checkPageBreak(45)

    doc.setFillColor(34, 197, 94) // Green
    doc.rect(margin, yPos, contentWidth, 8, 'F')
    doc.setTextColor(255, 255, 255)
    doc.setFontSize(12)
    doc.setFont('helvetica', 'bold')
    doc.text('DATASET STATISTICS', margin + 3, yPos + 5.5)
    yPos += 12

    doc.setTextColor(40, 40, 40)
    doc.setFontSize(10)
    doc.setFont('helvetica', 'normal')

    const datasetInfo = [
        ['Total Samples:', '3,264 brain MRI scans'],
        ['Training Set:', '2,286 samples (70%)'],
        ['Validation Set:', '489 samples (15%)'],
        ['Test Set:', '489 samples (15%)'],
        ['Classification:', 'Binary (Tumor / No Tumor)']
    ]

    datasetInfo.forEach(([label, value]) => {
        doc.setFont('helvetica', 'bold')
        doc.text(label, margin + 3, yPos)
        doc.setFont('helvetica', 'normal')
        doc.text(value, margin + 45, yPos)
        yPos += 5
    })
    yPos += 8

    // ========== AI DETAILED EXPLANATION SECTION ==========
    if (data.narrative) {
        checkPageBreak(60)

        doc.setFillColor(168, 85, 247) // Purple
        doc.rect(margin, yPos, contentWidth, 8, 'F')
        doc.setTextColor(255, 255, 255)
        doc.setFontSize(12)
        doc.setFont('helvetica', 'bold')
        doc.text('AI DETAILED EXPLANATION', margin + 3, yPos + 5.5)
        yPos += 12

        doc.setFontSize(9)
        doc.setTextColor(40, 40, 40)
        doc.setFont('helvetica', 'normal')

        // Parse narrative sections
        const sections = data.narrative.split('**').filter(s => s.trim())

        sections.forEach((section) => {
            checkPageBreak(20)

            const parts = section.split(':')
            if (parts.length > 1 && parts[0].length < 50) {
                // Section header
                doc.setFont('helvetica', 'bold')
                doc.setTextColor(88, 28, 135)
                yPos = addText(parts[0] + ':', margin + 3, yPos, { maxWidth: contentWidth - 6, lineHeight: 5 })
                yPos += 2

                // Section content
                doc.setFont('helvetica', 'normal')
                doc.setTextColor(50, 50, 50)
                yPos = addText(parts.slice(1).join(':').trim(), margin + 3, yPos, { maxWidth: contentWidth - 6, lineHeight: 5 })
                yPos += 5
            } else {
                // Regular paragraph
                doc.setFont('helvetica', 'normal')
                doc.setTextColor(50, 50, 50)
                yPos = addText(section.trim(), margin + 3, yPos, { maxWidth: contentWidth - 6, lineHeight: 5 })
                yPos += 5
            }
        })
        yPos += 5
    }

    // ========== LIMITATIONS SECTION ==========
    checkPageBreak(50)

    doc.setFillColor(239, 68, 68) // Red
    doc.rect(margin, yPos, contentWidth, 8, 'F')
    doc.setTextColor(255, 255, 255)
    doc.setFontSize(12)
    doc.setFont('helvetica', 'bold')
    doc.text('LIMITATIONS & RECOMMENDATIONS', margin + 3, yPos + 5.5)
    yPos += 12

    doc.setFillColor(254, 242, 242)
    doc.rect(margin, yPos, contentWidth, 40, 'F')
    doc.setDrawColor(239, 68, 68)
    doc.setLineWidth(0.5)
    doc.rect(margin, yPos, contentWidth, 40)

    doc.setFontSize(8)
    doc.setFont('helvetica', 'bold')
    doc.setTextColor(220, 38, 38)
    doc.text('IMPORTANT NOTICE:', margin + 3, yPos + 5)

    doc.setFont('helvetica', 'normal')
    doc.setTextColor(60, 60, 60)
    const limitText = 'This AI system is for research and educational use ONLY. Not a substitute for professional medical diagnosis. Binary classification without tumor type distinction. Accuracy depends on scan quality. No clinical validation or regulatory approval.'
    yPos = addText(limitText, margin + 3, yPos + 10, { maxWidth: contentWidth - 6, lineHeight: 4.5 })

    doc.setFont('helvetica', 'bold')
    doc.setTextColor(220, 38, 38)
    doc.text('RECOMMENDATION:', margin + 3, yPos + 5)
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(60, 60, 60)
    yPos = addText('Always consult qualified healthcare professionals for diagnosis. Use as supplementary screening only.', margin + 3, yPos + 9, { maxWidth: contentWidth - 6, lineHeight: 4.5 })
    yPos += 15

    // ========== FOOTER ==========
    const footerY = pageHeight - 15
    doc.setFillColor(248, 250, 252)
    doc.rect(0, footerY - 5, pageWidth, 20, 'F')

    doc.setDrawColor(200, 200, 200)
    doc.setLineWidth(0.3)
    doc.line(margin, footerY - 5, pageWidth - margin, footerY - 5)

    doc.setFontSize(7)
    doc.setTextColor(120, 120, 120)
    doc.setFont('helvetica', 'italic')
    doc.text('Generated by NeuroScan AI v1.0.0 - For research and educational use only', pageWidth / 2, footerY, { align: 'center' })
    doc.text('(c) 2025 NeuroScan AI - Brain Tumor Detection System', pageWidth / 2, footerY + 4, { align: 'center' })

    // Save PDF
    doc.save(`neuroscan-ai-report-${Date.now()}.pdf`)
}
