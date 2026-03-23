// ANETraining.swift — Native macOS Menu Bar App for ANE Training
// Single-file SwiftUI app using MenuBarExtra (.window style)
// Links against libane_train_stories110m.a via BridgingHeader.h

import SwiftUI
import Combine

// MARK: - App Entry Point

@main
struct ANETrainingApp: App {
    @StateObject private var trainingEngine = TrainingEngine()
    @StateObject private var generationEngine = GenerationEngine()
    @StateObject private var hardwareMonitor = HardwareMonitor()

    var body: some Scene {
        MenuBarExtra("ANE Training", systemImage: "brain") {
            ContentView()
                .environmentObject(trainingEngine)
                .environmentObject(generationEngine)
                .environmentObject(hardwareMonitor)
        }
        .menuBarExtraStyle(.window)
    }
}

// MARK: - Content View with Tabs

struct ContentView: View {
    @State private var selectedTab = 0

    var body: some View {
        VStack(spacing: 0) {
            // Tab bar
            HStack(spacing: 0) {
                TabButton(title: "Training", icon: "flame.fill", index: 0, selected: $selectedTab)
                TabButton(title: "Generate", icon: "text.cursor", index: 1, selected: $selectedTab)
                TabButton(title: "Monitor", icon: "gauge.with.dots.needle.33percent", index: 2, selected: $selectedTab)
            }
            .padding(.horizontal, 8)
            .padding(.top, 8)

            Divider().padding(.top, 4)

            // Tab content
            Group {
                switch selectedTab {
                case 0: TrainingView()
                case 1: GenerationView()
                case 2: MonitorView()
                default: TrainingView()
                }
            }
            .frame(width: 384, height: 440)
        }
        .frame(width: 400, height: 500)
    }
}

struct TabButton: View {
    let title: String
    let icon: String
    let index: Int
    @Binding var selected: Int

    var body: some View {
        Button(action: { selected = index }) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 11))
                Text(title)
                    .font(.system(size: 12, weight: .medium))
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(selected == index ? Color.accentColor.opacity(0.15) : Color.clear)
            .foregroundColor(selected == index ? .accentColor : .secondary)
            .cornerRadius(6)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Training Engine

class TrainingEngine: ObservableObject {
    @Published var isTraining = false
    @Published var step = 0
    @Published var loss: Float = 0
    @Published var lr: Float = 0
    @Published var msPerStep: Double = 0
    @Published var bestLoss: Float = Float.greatestFiniteMagnitude
    @Published var gradNorm: Float = 0
    @Published var fwdMs: Double = 0
    @Published var bwdMs: Double = 0
    @Published var updateMs: Double = 0
    @Published var errorMessage: String?

    private var session: OpaquePointer?
    private var shouldStop = false
    private let queue = DispatchQueue(label: "com.ane.training", qos: .userInitiated)

    func startTraining(dataPath: String, maxSteps: Int = 1000) {
        guard !isTraining else { return }

        DispatchQueue.main.async {
            self.isTraining = true
            self.shouldStop = false
            self.errorMessage = nil
            self.step = 0
            self.bestLoss = Float.greatestFiniteMagnitude
        }

        queue.async { [weak self] in
            guard let self = self else { return }

            var cfg = ane_train_default_config()
            cfg.max_steps = Int32(maxSteps)
            cfg.log_every = 0  // we handle logging ourselves

            var err = ANE_TRAIN_OK
            let session = ane_train_create(&cfg, &err)

            guard let session = session else {
                let msg = String(cString: ane_train_error_str(err))
                DispatchQueue.main.async {
                    self.errorMessage = "Create failed: \(msg)"
                    self.isTraining = false
                }
                return
            }

            self.session = session

            let dataErr = ane_train_load_data(session, dataPath)
            guard dataErr == ANE_TRAIN_OK else {
                let msg = String(cString: ane_train_error_str(dataErr))
                DispatchQueue.main.async {
                    self.errorMessage = "Load data failed: \(msg)"
                    self.isTraining = false
                }
                ane_train_destroy(session)
                self.session = nil
                return
            }

            // Set progress callback
            let selfPtr = Unmanaged.passUnretained(self).toOpaque()
            ane_train_set_progress_callback(session, trainingProgressCallback, selfPtr)

            // Run training
            let runErr = ane_train_run(session)
            if runErr != ANE_TRAIN_OK {
                let msg = String(cString: ane_train_error_str(runErr))
                DispatchQueue.main.async {
                    self.errorMessage = "Training error: \(msg)"
                }
            }

            DispatchQueue.main.async {
                self.isTraining = false
            }
        }
    }

    func stopTraining() {
        shouldStop = true
    }

    func saveCheckpoint(path: String) {
        guard let session = session else { return }
        let err = ane_train_save(session, path)
        if err != ANE_TRAIN_OK {
            let msg = String(cString: ane_train_error_str(err))
            DispatchQueue.main.async {
                self.errorMessage = "Save failed: \(msg)"
            }
        }
    }

    var shouldStopFlag: Bool { shouldStop }

    func updateFromResult(_ result: ANETrainStepResult) {
        DispatchQueue.main.async {
            self.step = Int(result.step)
            self.loss = result.loss
            self.lr = result.lr
            self.msPerStep = result.total_ms
            self.gradNorm = result.grad_norm
            self.fwdMs = result.fwd_ms
            self.bwdMs = result.bwd_ms
            self.updateMs = result.update_ms
            if result.loss < self.bestLoss {
                self.bestLoss = result.loss
            }
        }
    }

    deinit {
        if let session = session {
            ane_train_destroy(session)
        }
    }
}

// Top-level @convention(c) callback for training progress
private func trainingProgressCallback(
    _ result: UnsafePointer<ANETrainStepResult>?,
    _ ctx: UnsafeMutableRawPointer?
) -> Bool {
    guard let result = result, let ctx = ctx else { return false }
    let engine = Unmanaged<TrainingEngine>.fromOpaque(ctx).takeUnretainedValue()
    engine.updateFromResult(result.pointee)
    return !engine.shouldStopFlag
}

// MARK: - Generation Engine

class GenerationEngine: ObservableObject {
    @Published var isGenerating = false
    @Published var outputText = ""
    @Published var tokensGenerated = 0
    @Published var tokPerSec: Double = 0
    @Published var errorMessage: String?

    private let queue = DispatchQueue(label: "com.ane.generation", qos: .userInitiated)
    private var shouldStop = false

    func generate(prompt: String, checkpointPath: String, tokenizerPath: String, maxTokens: Int = 256) {
        guard !isGenerating else { return }

        DispatchQueue.main.async {
            self.isGenerating = true
            self.shouldStop = false
            self.outputText = ""
            self.tokensGenerated = 0
            self.tokPerSec = 0
            self.errorMessage = nil
        }

        queue.async { [weak self] in
            guard let self = self else { return }

            var err = ANE_TRAIN_OK
            let session = ane_gen_create(checkpointPath, tokenizerPath, &err)

            guard let session = session else {
                let msg = String(cString: ane_train_error_str(err))
                DispatchQueue.main.async {
                    self.errorMessage = "Gen create failed: \(msg)"
                    self.isGenerating = false
                }
                return
            }

            var cfg = ane_gen_default_config()
            cfg.max_tokens = Int32(maxTokens)

            let selfPtr = Unmanaged.passUnretained(self).toOpaque()
            let result = ane_gen_run(session, prompt, &cfg, generationTokenCallback, selfPtr)

            DispatchQueue.main.async {
                self.tokensGenerated = Int(result.tokens_generated)
                if result.ms_per_token > 0 {
                    self.tokPerSec = 1000.0 / result.ms_per_token
                }
                self.isGenerating = false
            }

            var mutableResult = result
            ane_gen_result_free(&mutableResult)
            ane_gen_destroy(session)
        }
    }

    func stopGenerating() {
        shouldStop = true
    }

    var shouldStopFlag: Bool { shouldStop }

    func appendToken(_ token: String) {
        DispatchQueue.main.async {
            self.outputText += token
        }
    }
}

// Top-level @convention(c) callback for token generation
private func generationTokenCallback(
    _ token: UnsafePointer<CChar>?,
    _ tokenId: Int32,
    _ ctx: UnsafeMutableRawPointer?
) -> Bool {
    guard let token = token, let ctx = ctx else { return false }
    let engine = Unmanaged<GenerationEngine>.fromOpaque(ctx).takeUnretainedValue()
    let str = String(cString: token)
    engine.appendToken(str)
    return !engine.shouldStopFlag
}

// MARK: - Hardware Monitor

class HardwareMonitor: ObservableObject {
    @Published var cpuUsage: Float = 0
    @Published var gpuUsage: Float = 0
    @Published var memUsedMB: Float = 0
    @Published var memTotalMB: Float = 0
    @Published var thermalState: Int = 0

    private var timer: Timer?

    init() {
        startMonitoring()
    }

    func startMonitoring() {
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.refresh()
        }
        refresh()
    }

    func stopMonitoring() {
        timer?.invalidate()
        timer = nil
    }

    func refresh() {
        let snap = ane_hw_snapshot()
        DispatchQueue.main.async {
            self.cpuUsage = snap.cpu_usage
            self.gpuUsage = snap.gpu_usage
            self.memUsedMB = Float(snap.mem_used_bytes) / (1024 * 1024)
            self.memTotalMB = Float(snap.mem_total_bytes) / (1024 * 1024)
            self.thermalState = Int(snap.thermal_state)
        }
    }

    var thermalLabel: String {
        switch thermalState {
        case 0: return "Nominal"
        case 1: return "Fair"
        case 2: return "Serious"
        case 3: return "Critical"
        default: return "Unknown"
        }
    }

    var thermalColor: Color {
        switch thermalState {
        case 0: return .green
        case 1: return .yellow
        case 2: return .orange
        case 3: return .red
        default: return .gray
        }
    }

    deinit {
        timer?.invalidate()
    }
}

// MARK: - Training View

struct TrainingView: View {
    @EnvironmentObject var engine: TrainingEngine
    @State private var dataPath = ""
    @State private var maxSteps = "1000"

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                // Model info card
                modelInfoCard

                Divider()

                // Training controls
                VStack(alignment: .leading, spacing: 8) {
                    Text("Training Data")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(.secondary)

                    HStack(spacing: 6) {
                        TextField("Path to data.bin", text: $dataPath)
                            .textFieldStyle(.roundedBorder)
                            .font(.system(size: 12))

                        Button(action: pickDataFile) {
                            Image(systemName: "folder")
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }

                    HStack {
                        Text("Max Steps:")
                            .font(.system(size: 12))
                        TextField("1000", text: $maxSteps)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 80)
                            .font(.system(size: 12))
                    }
                }

                // Start / Stop
                HStack {
                    if engine.isTraining {
                        Button(action: { engine.stopTraining() }) {
                            HStack(spacing: 4) {
                                Image(systemName: "stop.fill")
                                Text("Stop")
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.red)
                        .controlSize(.small)
                    } else {
                        Button(action: startTraining) {
                            HStack(spacing: 4) {
                                Image(systemName: "play.fill")
                                Text("Start Training")
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                        .disabled(dataPath.isEmpty)
                    }

                    Spacer()

                    if engine.isTraining {
                        ProgressView()
                            .scaleEffect(0.6)
                    }
                }

                // Error message
                if let err = engine.errorMessage {
                    HStack(spacing: 4) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.red)
                        Text(err)
                            .font(.system(size: 11))
                            .foregroundColor(.red)
                    }
                }

                // Live stats
                if engine.step > 0 {
                    Divider()
                    liveStatsView
                }

                Spacer(minLength: 0)
            }
            .padding(12)
        }
    }

    private var modelInfoCard: some View {
        let info = ane_train_model_info()
        let name = info.name != nil ? String(cString: info.name) : "Unknown"
        let params = Float(info.param_count) / 1_000_000.0

        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: "cpu")
                    .foregroundColor(.accentColor)
                Text(name)
                    .font(.system(size: 14, weight: .semibold))
                Spacer()
                Text(String(format: "%.1fM params", params))
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
            }

            HStack(spacing: 16) {
                StatPill(label: "Layers", value: "\(info.n_layers)")
                StatPill(label: "Dim", value: "\(info.dim)")
                StatPill(label: "Heads", value: "\(info.n_heads)")
                StatPill(label: "Vocab", value: "\(info.vocab_size)")
            }
        }
        .padding(10)
        .background(Color.primary.opacity(0.04))
        .cornerRadius(8)
    }

    private var liveStatsView: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Live Training")
                .font(.system(size: 11, weight: .semibold))
                .foregroundColor(.secondary)

            HStack(spacing: 16) {
                StatBlock(label: "Step", value: "\(engine.step)", icon: "number")
                StatBlock(label: "Loss", value: String(format: "%.4f", engine.loss), icon: "chart.line.downtrend.xyaxis")
                StatBlock(label: "Best", value: engine.bestLoss < Float.greatestFiniteMagnitude
                          ? String(format: "%.4f", engine.bestLoss) : "--", icon: "star.fill")
            }

            HStack(spacing: 16) {
                StatBlock(label: "ms/step", value: String(format: "%.1f", engine.msPerStep), icon: "clock")
                StatBlock(label: "LR", value: String(format: "%.2e", engine.lr), icon: "dial.low")
                StatBlock(label: "Grad", value: String(format: "%.2f", engine.gradNorm), icon: "arrow.up.right")
            }

            // Timing breakdown bar
            timingBar
        }
    }

    private var timingBar: some View {
        let total = max(engine.fwdMs + engine.bwdMs + engine.updateMs, 0.001)
        let fwdPct = engine.fwdMs / total
        let bwdPct = engine.bwdMs / total
        let updPct = engine.updateMs / total

        return VStack(alignment: .leading, spacing: 4) {
            Text("Timing")
                .font(.system(size: 10, weight: .medium))
                .foregroundColor(.secondary)

            GeometryReader { geo in
                HStack(spacing: 1) {
                    Rectangle()
                        .fill(Color.blue)
                        .frame(width: geo.size.width * fwdPct)
                    Rectangle()
                        .fill(Color.orange)
                        .frame(width: geo.size.width * bwdPct)
                    Rectangle()
                        .fill(Color.green)
                        .frame(width: geo.size.width * updPct)
                }
                .cornerRadius(3)
            }
            .frame(height: 8)

            HStack(spacing: 12) {
                LegendDot(color: .blue, label: String(format: "Fwd %.0fms", engine.fwdMs))
                LegendDot(color: .orange, label: String(format: "Bwd %.0fms", engine.bwdMs))
                LegendDot(color: .green, label: String(format: "Upd %.0fms", engine.updateMs))
            }
            .font(.system(size: 10))
        }
    }

    private func startTraining() {
        let steps = Int(maxSteps) ?? 1000
        engine.startTraining(dataPath: dataPath, maxSteps: steps)
    }

    private func pickDataFile() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.data]
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            dataPath = url.path
        }
    }
}

// MARK: - Generation View

struct GenerationView: View {
    @EnvironmentObject var engine: GenerationEngine
    @State private var prompt = "Once upon a time"
    @State private var checkpointPath = ""
    @State private var tokenizerPath = ""
    @State private var maxTokens = "256"

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                // Paths
                VStack(alignment: .leading, spacing: 6) {
                    Text("Configuration")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(.secondary)

                    PathField(label: "Checkpoint", path: $checkpointPath)
                    PathField(label: "Tokenizer", path: $tokenizerPath)

                    HStack {
                        Text("Max Tokens:")
                            .font(.system(size: 12))
                        TextField("256", text: $maxTokens)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 60)
                            .font(.system(size: 12))
                    }
                }

                Divider()

                // Prompt
                VStack(alignment: .leading, spacing: 4) {
                    Text("Prompt")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(.secondary)

                    TextField("Enter prompt...", text: $prompt)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 12))
                }

                // Generate button
                HStack {
                    if engine.isGenerating {
                        Button(action: { engine.stopGenerating() }) {
                            HStack(spacing: 4) {
                                Image(systemName: "stop.fill")
                                Text("Stop")
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.red)
                        .controlSize(.small)
                    } else {
                        Button(action: startGeneration) {
                            HStack(spacing: 4) {
                                Image(systemName: "sparkles")
                                Text("Generate")
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                        .disabled(checkpointPath.isEmpty || tokenizerPath.isEmpty)
                    }

                    Spacer()

                    if engine.tokensGenerated > 0 {
                        Text("\(engine.tokensGenerated) tok")
                            .font(.system(size: 11))
                            .foregroundColor(.secondary)
                        if engine.tokPerSec > 0 {
                            Text(String(format: "%.1f tok/s", engine.tokPerSec))
                                .font(.system(size: 11, weight: .medium))
                                .foregroundColor(.accentColor)
                        }
                    }
                }

                // Error
                if let err = engine.errorMessage {
                    HStack(spacing: 4) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.red)
                        Text(err)
                            .font(.system(size: 11))
                            .foregroundColor(.red)
                    }
                }

                // Output
                if !engine.outputText.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Output")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundColor(.secondary)

                        Text(engine.outputText)
                            .font(.system(size: 12, design: .monospaced))
                            .textSelection(.enabled)
                            .padding(8)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.primary.opacity(0.04))
                            .cornerRadius(6)
                    }
                }

                Spacer(minLength: 0)
            }
            .padding(12)
        }
    }

    private func startGeneration() {
        let tokens = Int(maxTokens) ?? 256
        engine.generate(
            prompt: prompt,
            checkpointPath: checkpointPath,
            tokenizerPath: tokenizerPath,
            maxTokens: tokens
        )
    }
}

// MARK: - Monitor View

struct MonitorView: View {
    @EnvironmentObject var monitor: HardwareMonitor

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Header
            HStack {
                Text("Hardware Monitor")
                    .font(.system(size: 14, weight: .semibold))
                Spacer()
                // Thermal badge
                HStack(spacing: 4) {
                    Image(systemName: "thermometer.medium")
                        .font(.system(size: 11))
                    Text(monitor.thermalLabel)
                        .font(.system(size: 11, weight: .medium))
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(monitor.thermalColor.opacity(0.15))
                .foregroundColor(monitor.thermalColor)
                .cornerRadius(10)
            }

            // CPU gauge
            GaugeRow(
                label: "CPU",
                value: monitor.cpuUsage,
                maxValue: 100,
                unit: "%",
                icon: "cpu",
                color: .blue
            )

            // GPU gauge
            GaugeRow(
                label: "GPU",
                value: monitor.gpuUsage,
                maxValue: 100,
                unit: "%",
                icon: "display",
                color: .green
            )

            // Memory bar
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Image(systemName: "memorychip")
                        .font(.system(size: 12))
                        .foregroundColor(.purple)
                    Text("Memory")
                        .font(.system(size: 12, weight: .medium))
                    Spacer()
                    Text(String(format: "%.0f / %.0f MB",
                                monitor.memUsedMB, monitor.memTotalMB))
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(.secondary)
                }

                let memPct = monitor.memTotalMB > 0
                    ? min(monitor.memUsedMB / monitor.memTotalMB, 1.0)
                    : Float(0)

                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color.primary.opacity(0.08))
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color.purple.opacity(0.6))
                            .frame(width: geo.size.width * CGFloat(memPct))
                    }
                }
                .frame(height: 10)
            }

            // Device info
            Divider()
            VStack(alignment: .leading, spacing: 4) {
                Text("ANE Device")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(.secondary)

                let info = ane_train_model_info()
                let name = info.name != nil ? String(cString: info.name) : "Unknown"
                HStack {
                    Text("Model:")
                        .foregroundColor(.secondary)
                    Text(name)
                }
                .font(.system(size: 11))

                HStack {
                    Text("ANE Compile Budget:")
                        .foregroundColor(.secondary)
                    Text("~119 kernels/process")
                }
                .font(.system(size: 11))

                HStack {
                    Text("SRAM:")
                        .foregroundColor(.secondary)
                    Text("~32 MB")
                }
                .font(.system(size: 11))
            }

            Spacer()

            // Quit button
            HStack {
                Spacer()
                Button(action: { NSApp.terminate(nil) }) {
                    HStack(spacing: 4) {
                        Image(systemName: "power")
                        Text("Quit")
                    }
                    .font(.system(size: 11))
                }
                .buttonStyle(.plain)
                .foregroundColor(.secondary)
            }
        }
        .padding(12)
    }
}

// MARK: - Reusable Components

struct StatPill: View {
    let label: String
    let value: String

    var body: some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
            Text(label)
                .font(.system(size: 9))
                .foregroundColor(.secondary)
        }
    }
}

struct StatBlock: View {
    let label: String
    let value: String
    let icon: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 3) {
                Image(systemName: icon)
                    .font(.system(size: 9))
                    .foregroundColor(.secondary)
                Text(label)
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
            }
            Text(value)
                .font(.system(size: 13, weight: .medium, design: .monospaced))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct LegendDot: View {
    let color: Color
    let label: String

    var body: some View {
        HStack(spacing: 3) {
            Circle()
                .fill(color)
                .frame(width: 6, height: 6)
            Text(label)
                .foregroundColor(.secondary)
        }
    }
}

struct GaugeRow: View {
    let label: String
    let value: Float
    let maxValue: Float
    let unit: String
    let icon: String
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: icon)
                    .font(.system(size: 12))
                    .foregroundColor(color)
                Text(label)
                    .font(.system(size: 12, weight: .medium))
                Spacer()
                Text(String(format: "%.1f%@", value, unit))
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.secondary)
            }

            let pct = maxValue > 0 ? min(value / maxValue, 1.0) : Float(0)

            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.primary.opacity(0.08))
                    RoundedRectangle(cornerRadius: 4)
                        .fill(color.opacity(0.6))
                        .frame(width: geo.size.width * CGFloat(pct))
                }
            }
            .frame(height: 10)
        }
    }
}

struct PathField: View {
    let label: String
    @Binding var path: String

    var body: some View {
        HStack(spacing: 6) {
            Text(label + ":")
                .font(.system(size: 11))
                .frame(width: 72, alignment: .trailing)
                .foregroundColor(.secondary)
            TextField("path/to/file", text: $path)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 11))
            Button(action: pickFile) {
                Image(systemName: "folder")
            }
            .buttonStyle(.bordered)
            .controlSize(.mini)
        }
    }

    private func pickFile() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            path = url.path
        }
    }
}
