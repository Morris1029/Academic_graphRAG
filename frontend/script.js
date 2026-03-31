// Global variables
let selectedFiles = [];
let datasets = [];
let currentDataset = null;
let graphChart = null;
let qaSubgraphChart = null;
// let queryChart = null;
let questionCount = 0;

// i18n configuration: default to Chinese for this project
let currentLang = 'zh';
const i18n = {
    en: {
        processing: 'Processing your research question...',
        pleaseSelectAndEnter: 'Please select a dataset and enter a question',
        subtitle: 'Interdisciplinary knowledge discovery with Graph Retrieval-Augmented Generation',
        tabs: { dashboard: 'Overview', upload: 'Data Upload', graph: 'Graph Visualization', qa: 'Research Q&A' },
        systemOverview: 'System Overview',
        statDatasetsLabel: 'Datasets',
        statGraphsLabel: 'Constructed Graphs',
        statQuestionsLabel: 'Questions Asked',
        systemStatusLabel: 'System Status',
        systemStatusConnected: 'Connected',
        researchInsightTitle: 'Research Orientation',
        researchRouteTitle: 'Research Route',
        researchRouteDesc: 'Knowledge graph construction -> GraphRAG retrieval and reasoning -> evaluation loop',
        researchEvalTitle: 'Evaluation Metrics',
        researchEvalDesc: 'Answer accuracy, reasoning explainability, interdisciplinary association quality',
        researchPlanTitle: 'Stage Plan',
        researchPlanDesc: 'Data and graph / system experiments / evaluation and thesis writing',
        quickActions: 'Quick Actions',
        qaUploadDocsBtn: 'Upload Documents',
        qaViewGraphBtn: 'View Graph',
        qaQueryBtn: 'Start Q&A',
        qaRefreshBtn: 'Refresh Data',
        uploadDocuments: 'Upload Documents',
        uploadDragTitle: 'Drag and drop files here',
        uploadDragDesc: 'or click to browse files',
        uploadSupports: 'Currently supports: .txt, .md, .json, .pdf, .docx, .doc',
        uploadSampleHeader: 'Sample Format (.json):',
        uploadBtn: 'Upload Files',
        clearUploadBtn: 'Clear',
        availableDatasets: 'Available Datasets',
        knowledgeTreeTitle: 'Knowledge Graph Visualization',
        graphHintText: 'Tip: Dark graph panels emphasize interpretable paths and relation structures.',
        labelDataset: 'Dataset:',
        selectDatasetPlaceholder: 'Select a dataset...',
        queryPanelTitle: 'Research Q&A Panel',
        labelQuery: 'Research Question:',
        qaHintText: 'Example: What are the shared and different application paths of LLM technology across disciplines?',
        questionPlaceholder: 'Enter a research question for interdisciplinary knowledge discovery...',
        askBtn: 'Start Q&A',
        answerTitle: 'Answer:',
        uploadProgress: (p) => `Uploading... ${p}%`,
        uploadCompleted: 'Upload completed!',
        uploadFailed: 'Upload failed: ',
        loadDatasetFailed: 'Failed to load datasets',
        noDatasets: 'No datasets available. Upload files to start graph construction.',
        schemaTag: 'Schema',
        schemaCustom: 'custom',
        schemaDefault: 'default',
        datasetStatusMap: {
            ready: 'ready',
            constructing: 'constructing',
            reconstructing: 'reconstructing',
            needs_construction: 'needs construction',
            error: 'error'
        },
        btnConstruct: 'Construct',
        btnReconstruct: 'Reconstruct',
        btnUploadSchema: 'Upload Schema',
        btnDelete: 'Delete',
        msgUploadingSchema: (name) => `Uploading schema for "${name}"...`,
        msgSchemaUploaded: 'Schema uploaded successfully!',
        msgSchemaUploadFailed: 'Schema upload failed: ',
        msgStartConstruct: 'Starting graph construction...',
        msgConstructProgress: (idx, msg) => `[Construct ${idx}] ${msg}`,
        msgConstructCompleted: 'Graph construction completed!',
        msgConstructError: (msg) => `Construction error: ${msg}`,
        msgConstructFailed: 'Graph construction failed: ',
        confirmReconstruct: (name) => `Are you sure you want to reconstruct "${name}"? Existing graph and cache files will be removed.`,
        msgStartReconstruct: 'Reconstructing graph...',
        msgReconstructProgress: (idx, msg) => `[Reconstruct ${idx}] ${msg}`,
        msgReconstructCompleted: 'Graph reconstruction completed!',
        msgReconstructError: (msg) => `Reconstruction error: ${msg}`,
        msgReconstructFailed: 'Graph reconstruction failed: ',
        confirmDelete: (name) => `Are you sure you want to delete dataset "${name}"? This action cannot be undone.`,
        msgDeletingDataset: 'Deleting dataset...',
        msgDatasetDeleted: (name) => `Dataset "${name}" deleted successfully!`,
        msgDeleteFailed: 'Dataset deletion failed: ',
        msgLoadGraphFailed: 'Failed to load graph data',
        msgQuestionFailed: 'Failed to process question: ',
        previewPrefix: 'Preview: ',
        decomposeFallbackNotice: 'Decomposition fallback enabled due to schema decode/parse issue; retrieval may be less precise.',
        decomposeStart: 'Starting to decompose your question into sub-questions...',
        decomposeSummary: (t) => `Decomposition complete. Total sub-questions: ${t}`,
        subQuestion: (i,t,q) => `Sub-question ${i}/${t}: ${q}`,
        ircotStart: 'Starting IRCoT reasoning...',
        ircotStep: (s,max,thought) => `IRCoT Step ${s}/${max}: ${thought}`,
        retrievalProgress: (found,cand) => `Retrieval progress: ${found} relevant nodes from ${cand} candidates.`,
        answerStart: 'Synthesizing the final answer...',
        qaCompleted: 'Q&A completed',
        completedBadge: 'Completed',
        retrievalStatsTitle: 'Retrieval Statistics',
        retrievalSubQuestions: 'Sub-questions:',
        retrievalTriplesTotal: 'Retrieved triples (sum):',
        retrievalChunks: 'Relevant chunks:',
        questionDecompositionTitle: 'Question Decomposition',
        retrievedTriplesLabel: 'Retrieved triples:',
        retrievedChunksLabel: 'Retrieved chunks:',
        triplesLabel: 'Triples',
        chunksLabel: 'Chunks',
        timeLabel: 'Time',
        noSubQuestionData: 'No sub-question data',
        subgraphTitle: 'Subgraph Visualization',
        retrievedTriplesSummary: (n) => `Retrieved ${n} related triples`,
        moreTriples: (n) => `...and ${n} more`,
        showLess: 'Show less',
        echartsNotLoaded: 'ECharts library not loaded',
        echartsInitError: 'Error initializing ECharts: ',
        noEntityRelationships: 'No entity relationships found',
        chartRenderError: 'Chart rendering error: ',
        nodeTypeLabel: 'Type',
        relationLabel: 'Relation',
        datasetDocCountLabel: (count) => count === 1 ? ' (1 document)' : ` (${count} documents)`,
        uploadSuccessMessage: (name, count) => {
            const friendlyName = (name || '').replace(/_/g, ' ').trim() || name || 'dataset';
            if (typeof count === 'number' && count > 0) {
                return `Dataset "${friendlyName}" uploaded successfully${count === 1 ? ' (1 document)' : ` (${count} documents)`}!`;
            }
            return `Dataset "${friendlyName}" uploaded successfully!`;
        }
    },
    zh: {
        processing: '正在处理你的研究问题...',
        pleaseSelectAndEnter: '请选择数据集并输入问题',
        subtitle: '面向跨学科知识发现的图检索增强生成研究平台',
        tabs: { dashboard: '概览', upload: '数据上传', graph: '图谱可视化', qa: '研究问答' },
        systemOverview: '系统概览',
        statDatasetsLabel: '数据集数量',
        statGraphsLabel: '已构建图谱',
        statQuestionsLabel: '问答次数',
        systemStatusLabel: '系统状态',
        systemStatusConnected: '已连接',
        researchInsightTitle: '研究导向',
        researchRouteTitle: '研究路线',
        researchRouteDesc: '知识图谱构建 -> GraphRAG 检索推理 -> 评估闭环',
        researchEvalTitle: '评估指标',
        researchEvalDesc: '答案准确性、推理可解释性、跨学科关联质量',
        researchPlanTitle: '阶段进度',
        researchPlanDesc: '数据与图谱 / 系统实验 / 评估与论文写作',
        quickActions: '快捷操作',
        qaUploadDocsBtn: '上传文档',
        qaViewGraphBtn: '查看图谱',
        qaQueryBtn: '发起问答',
        qaRefreshBtn: '刷新数据',
        uploadDocuments: '上传文档',
        uploadDragTitle: '拖拽文件到这里',
        uploadDragDesc: '或点击选择文件',
        uploadSupports: '当前支持：.txt、.md、.json、.pdf、.docx、.doc',
        uploadSampleHeader: '📄 示例格式（.json）：',
        uploadBtn: '开始上传',
        clearUploadBtn: '清空',
        availableDatasets: '可用数据集',
        knowledgeTreeTitle: '知识图谱可视化',
        graphHintText: '提示：深色图谱面板用于强化可解释路径，支持节点关系与子图结构观察。',
        labelDataset: '数据集：',
        selectDatasetPlaceholder: '请选择数据集…',
        queryPanelTitle: '研究问答面板',
        labelQuery: '研究问题：',
        qaHintText: '示例：大语言模型技术在不同学科中的应用路径有哪些共性与差异？',
        questionPlaceholder: '请输入研究问题，支持复杂推理与跨学科关联检索...',
        askBtn: '开始问答',
        answerTitle: '答案：',
        uploadProgress: (p) => `上传中... ${p}%`,
        uploadCompleted: '上传完成！',
        uploadFailed: '上传失败：',
        loadDatasetFailed: '加载数据集失败',
        noDatasets: '暂无可用数据集，请先上传文件。',
        schemaTag: 'Schema',
        schemaCustom: '自定义',
        schemaDefault: '默认',
        datasetStatusMap: {
            ready: '就绪',
            constructing: '构建中',
            reconstructing: '重建中',
            needs_construction: '待构建',
            error: '错误'
        },
        btnConstruct: '构建图谱',
        btnReconstruct: '重建图谱',
        btnUploadSchema: '上传 Schema',
        btnDelete: '删除',
        msgUploadingSchema: (name) => `正在为“${name}”上传 Schema...`,
        msgSchemaUploaded: 'Schema 上传成功！',
        msgSchemaUploadFailed: 'Schema 上传失败：',
        msgStartConstruct: '开始构建图谱...',
        msgConstructProgress: (idx, msg) => `【构建 ${idx}】${msg}`,
        msgConstructCompleted: '图谱构建完成！',
        msgConstructError: (msg) => `构建错误：${msg}`,
        msgConstructFailed: '图谱构建失败：',
        confirmReconstruct: (name) => `确认重建数据集“${name}”的图谱吗？这将删除已有图谱及缓存文件。`,
        msgStartReconstruct: '开始重建图谱...',
        msgReconstructProgress: (idx, msg) => `【重建 ${idx}】${msg}`,
        msgReconstructCompleted: '图谱重建完成！',
        msgReconstructError: (msg) => `重建错误：${msg}`,
        msgReconstructFailed: '图谱重建失败：',
        confirmDelete: (name) => `确认删除数据集“${name}”吗？该操作不可恢复。`,
        msgDeletingDataset: '正在删除数据集...',
        msgDatasetDeleted: (name) => `数据集“${name}”删除成功！`,
        msgDeleteFailed: '数据集删除失败：',
        msgLoadGraphFailed: '图谱数据加载失败',
        msgQuestionFailed: '问题处理失败：',
        previewPrefix: '预览：',
        decomposeFallbackNotice: '问题分解已降级（Schema 解码/解析异常），当前检索精度可能下降。',
        decomposeStart: '开始将你的问题分解为子问题…',
        decomposeSummary: (t) => `分解完成。子问题总数：${t}`,
        subQuestion: (i,t,q) => `子问题 ${i}/${t}：${q}`,
        ircotStart: '开始 IRCoT 推理…',
        ircotStep: (s,max,thought) => `IRCoT 步骤 ${s}/${max}：${thought}`,
        retrievalProgress: (found,cand) => `检索进度：在 ${cand} 个候选中找到 ${found} 个相关节点。`,
        answerStart: '正在综合生成最终答案…',
        qaCompleted: '问答完成',
        completedBadge: '已完成',
        retrievalStatsTitle: '检索统计',
        retrievalSubQuestions: '子问题数量：',
        retrievalTriplesTotal: '检索三元组总数：',
        retrievalChunks: '相关文本片段：',
        questionDecompositionTitle: '问题分解',
        retrievedTriplesLabel: '检索三元组：',
        retrievedChunksLabel: '检索文本块：',
        triplesLabel: '三元组',
        chunksLabel: '片段',
        timeLabel: '耗时',
        noSubQuestionData: '暂无子问题数据',
        subgraphTitle: '子图可视化',
        retrievedTriplesSummary: (n) => `共检索到 ${n} 条相关三元组`,
        moreTriples: (n) => `...还有 ${n} 条`,
        showLess: '收起',
        echartsNotLoaded: 'ECharts 组件未加载',
        echartsInitError: 'ECharts 初始化失败：',
        noEntityRelationships: '未发现可展示的实体关系',
        chartRenderError: '图表渲染错误：',
        nodeTypeLabel: '类型',
        relationLabel: '关系',
        datasetDocCountLabel: (count) => count === 1 ? '（1篇文档）' : `（共${count}篇文档）`,
        uploadSuccessMessage: (name, count) => {
            const friendlyName = (name || '').replace(/_/g, ' ').trim() || name || '数据集';
            if (typeof count === 'number' && count > 0) {
                return `数据集“${friendlyName}”上传成功${count === 1 ? '（1篇文档）' : `（共${count}篇文档）`}！`;
            }
            return `数据集“${friendlyName}”上传成功！`;
        }
    }
};

function refreshUITexts() {
    const t = i18n[currentLang];
    // Header & tabs
    const subtitle = document.getElementById('subtitleText'); if (subtitle) subtitle.textContent = t.subtitle;
    const tabDashboard = document.getElementById('tabDashboard'); if (tabDashboard) tabDashboard.textContent = t.tabs.dashboard;
    const tabUpload = document.getElementById('tabUpload'); if (tabUpload) tabUpload.textContent = t.tabs.upload;
    const tabGraph = document.getElementById('tabGraph'); if (tabGraph) tabGraph.textContent = t.tabs.graph;
    const tabQA = document.getElementById('tabQA'); if (tabQA) tabQA.textContent = t.tabs.qa;
    // Dashboard
    const sysTitle = document.getElementById('systemOverviewTitle'); if (sysTitle) sysTitle.textContent = t.systemOverview;
    const statDatasetsLabel = document.getElementById('statDatasetsLabel'); if (statDatasetsLabel) statDatasetsLabel.textContent = t.statDatasetsLabel;
    const statGraphsLabel = document.getElementById('statGraphsLabel'); if (statGraphsLabel) statGraphsLabel.textContent = t.statGraphsLabel;
    const statQuestionsLabel = document.getElementById('statQuestionsLabel'); if (statQuestionsLabel) statQuestionsLabel.textContent = t.statQuestionsLabel;
    const systemStatusLabel = document.getElementById('systemStatusLabel'); if (systemStatusLabel) systemStatusLabel.textContent = t.systemStatusLabel;
    const systemStatusBadge = document.getElementById('systemStatusBadge'); if (systemStatusBadge) systemStatusBadge.textContent = t.systemStatusConnected;
    const researchInsightTitle = document.getElementById('researchInsightTitle'); if (researchInsightTitle) researchInsightTitle.textContent = t.researchInsightTitle;
    const researchRouteTitle = document.getElementById('researchRouteTitle'); if (researchRouteTitle) researchRouteTitle.textContent = t.researchRouteTitle;
    const researchRouteDesc = document.getElementById('researchRouteDesc'); if (researchRouteDesc) researchRouteDesc.textContent = t.researchRouteDesc;
    const researchEvalTitle = document.getElementById('researchEvalTitle'); if (researchEvalTitle) researchEvalTitle.textContent = t.researchEvalTitle;
    const researchEvalDesc = document.getElementById('researchEvalDesc'); if (researchEvalDesc) researchEvalDesc.textContent = t.researchEvalDesc;
    const researchPlanTitle = document.getElementById('researchPlanTitle'); if (researchPlanTitle) researchPlanTitle.textContent = t.researchPlanTitle;
    const researchPlanDesc = document.getElementById('researchPlanDesc'); if (researchPlanDesc) researchPlanDesc.textContent = t.researchPlanDesc;
    const qaTitle = document.getElementById('quickActionsTitle'); if (qaTitle) qaTitle.textContent = t.quickActions;
    const qaUploadDocsBtn = document.getElementById('qaUploadDocsBtn'); if (qaUploadDocsBtn) qaUploadDocsBtn.textContent = t.qaUploadDocsBtn;
    const qaViewGraphBtn = document.getElementById('qaViewGraphBtn'); if (qaViewGraphBtn) qaViewGraphBtn.textContent = t.qaViewGraphBtn;
    const qaQueryBtn = document.getElementById('qaQueryBtn'); if (qaQueryBtn) qaQueryBtn.textContent = t.qaQueryBtn;
    const qaRefreshBtn = document.getElementById('qaRefreshBtn'); if (qaRefreshBtn) qaRefreshBtn.textContent = t.qaRefreshBtn;
    // Upload
    const uploadTitle = document.getElementById('uploadDocumentsTitle'); if (uploadTitle) uploadTitle.textContent = t.uploadDocuments;
    const uploadDragTitle = document.getElementById('uploadDragTitle'); if (uploadDragTitle) uploadDragTitle.textContent = t.uploadDragTitle;
    const uploadDragDesc = document.getElementById('uploadDragDesc'); if (uploadDragDesc) uploadDragDesc.textContent = t.uploadDragDesc;
    const uploadSupports = document.getElementById('uploadSupports'); if (uploadSupports) uploadSupports.textContent = t.uploadSupports;
    const uploadSampleHeader = document.getElementById('uploadSampleHeader'); if (uploadSampleHeader) uploadSampleHeader.textContent = t.uploadSampleHeader;
    const uploadBtn = document.getElementById('uploadBtn'); if (uploadBtn) uploadBtn.textContent = t.uploadBtn;
    const clearUploadBtn = document.getElementById('clearUploadBtn'); if (clearUploadBtn) clearUploadBtn.textContent = t.clearUploadBtn;
    const datasetsTitle = document.getElementById('availableDatasetsTitle'); if (datasetsTitle) datasetsTitle.textContent = t.availableDatasets;
    // Graph
    const knowledgeTreeTitle = document.getElementById('knowledgeTreeTitle'); if (knowledgeTreeTitle) knowledgeTreeTitle.textContent = t.knowledgeTreeTitle;
    const graphHintText = document.getElementById('graphHintText'); if (graphHintText) graphHintText.textContent = t.graphHintText;
    const labelGraphDataset = document.getElementById('labelGraphDataset'); if (labelGraphDataset) labelGraphDataset.textContent = t.labelDataset;
    const graphSelect = document.getElementById('graphDataset'); if (graphSelect) graphSelect.options[0].text = t.selectDatasetPlaceholder;
    // QA
    const queryPanelTitle = document.getElementById('queryPanelTitle'); if (queryPanelTitle) queryPanelTitle.textContent = t.queryPanelTitle;
    const labelQADataset = document.getElementById('labelQADataset'); if (labelQADataset) labelQADataset.textContent = t.labelDataset;
    const qaSelect = document.getElementById('qaDataset'); if (qaSelect) qaSelect.options[0].text = t.selectDatasetPlaceholder;
    const labelQuery = document.getElementById('labelQuery'); if (labelQuery) labelQuery.textContent = t.labelQuery;
    const qaHintText = document.getElementById('qaHintText'); if (qaHintText) qaHintText.textContent = t.qaHintText;
    const questionInput = document.getElementById('questionInput'); if (questionInput) questionInput.placeholder = t.questionPlaceholder;
    const askBtn = document.getElementById('askBtn'); if (askBtn) askBtn.textContent = t.askBtn;
    const answerTitle = document.getElementById('answerTitle'); if (answerTitle) answerTitle.textContent = t.answerTitle;
    // Processing text & badge (if visible)
    const procText = document.getElementById('qaProcessingText'); if (procText) procText.textContent = t.processing;
    const badge = document.getElementById('qaCompletionBadge'); if (badge && badge.style.display !== 'none') badge.textContent = t.completedBadge;

    // Refresh dataset renderings to reflect language-specific labels
    displayDatasets();
    loadDatasetOptions('graphDataset');
    loadDatasetOptions('qaDataset');
}

// API base URL
const API_BASE = '';

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
});

function initializeApp() {
    refreshData();
    console.log('Youtu-GraphRAG initialized');
    // Apply initial language to UI
    refreshUITexts();
}

function setupEventListeners() {
    // File upload
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');

    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);

    // Question input
    document.getElementById('questionInput').addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            askQuestion();
        }
    });

    // initialize language buttons state
    updateLangButtons();
}

function setLanguage(lang) {
    if (lang !== 'en' && lang !== 'zh') return;
    currentLang = lang;
    updateLangButtons();
    // Update visible texts
    refreshUITexts();
}

function updateLangButtons() {
    const enBtn = document.getElementById('langEnBtn');
    const zhBtn = document.getElementById('langZhBtn');
    if (!enBtn || !zhBtn) return;
    enBtn.classList.toggle('active', currentLang === 'en');
    zhBtn.classList.toggle('active', currentLang === 'zh');
}

function switchTab(tabName, tabEl) {
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    const targetTab = tabEl || document.querySelector(`[onclick*="switchTab('${tabName}'"]`);
    if (targetTab) targetTab.classList.add('active');

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    document.getElementById(tabName).classList.add('active');

    // Load data for specific tabs
    if (tabName === 'upload') {
        loadDatasets();
    } else if (tabName === 'graph') {
        loadDatasetOptions('graphDataset');
    } else if (tabName === 'qa') {
        loadDatasetOptions('qaDataset');
    }
}

// File handling
function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files);
    addFiles(files);
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    addFiles(files);
}

function addFiles(files) {
    selectedFiles = [...selectedFiles, ...files];
    updateFileList();
    document.getElementById('uploadBtn').disabled = selectedFiles.length === 0;
}

function updateFileList() {
    const fileList = document.getElementById('fileList');
    if (selectedFiles.length === 0) {
        fileList.classList.add('hidden');
        return;
    }

    fileList.classList.remove('hidden');
    fileList.innerHTML = selectedFiles.map((file, index) => `
        <div class="file-item">
            <span>📄 ${file.name}</span>
            <span>${formatFileSize(file.size)} <button onclick="removeFile(${index})" style="background:none;border:none;color:#ff4d4f;cursor:pointer;">✕</button></span>
        </div>
    `).join('');
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    updateFileList();
    document.getElementById('uploadBtn').disabled = selectedFiles.length === 0;
}

function clearFiles() {
    selectedFiles = [];
    document.getElementById('fileInput').value = '';
    updateFileList();
    document.getElementById('uploadBtn').disabled = true;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDatasetDisplayName(dataset) {
    if (!dataset) return '';
    const rawName = (dataset.display_label || dataset.name || '').toString();
    const cleanedName = rawName.replace(/_/g, ' ').trim() || (dataset.name || '');
    const docCount = typeof dataset.docs_count === 'number' ? dataset.docs_count : null;
    if (docCount && docCount > 0) {
        const langConfig = i18n[currentLang] || i18n.en;
        const suffixFn = langConfig?.datasetDocCountLabel;
        const suffix = suffixFn ? suffixFn(docCount) : ` (${docCount} documents)`;
        return `${cleanedName}${suffix}`;
    }
    return cleanedName;
}

async function uploadFiles() {
    if (selectedFiles.length === 0) return;

    const formData = new FormData();
    selectedFiles.forEach(file => formData.append('files', file));
    formData.append('client_id', 'web_client');

    const progressSection = document.getElementById('uploadProgress');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');

    progressSection.classList.remove('hidden');
    document.getElementById('uploadBtn').disabled = true;

    try {
        // Simulate progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 20;
            if (progress > 90) progress = 90;
            progressFill.style.width = progress + '%';
            progressText.textContent = (i18n[currentLang] || i18n.en).uploadProgress(Math.round(progress));
        }, 500);

        const response = await axios.post(`${API_BASE}/api/upload`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });

        clearInterval(progressInterval);
        progressFill.style.width = '100%';
        progressText.textContent = (i18n[currentLang] || i18n.en).uploadCompleted;

        const uploadData = response?.data || {};
        const datasetName = uploadData.dataset_name;
        const filesCount = uploadData.files_count;
        const langConfig = i18n[currentLang] || i18n.en;
        if (langConfig?.uploadSuccessMessage) {
            showMessage(langConfig.uploadSuccessMessage(datasetName, filesCount), 'success');
        } else {
            let successMessage = datasetName ? `Dataset "${datasetName}" uploaded successfully!` : 'Files uploaded successfully!';
            if (typeof filesCount === 'number' && filesCount > 0) {
                successMessage += filesCount === 1 ? ' (1 document)' : ` (${filesCount} documents)`;
            }
            showMessage(successMessage, 'success');
        }
        clearFiles();
        refreshData();

        setTimeout(() => {
            progressSection.classList.add('hidden');
        }, 2000);

    } catch (error) {
        console.error('Upload failed:', error);
        const t = i18n[currentLang] || i18n.en;
        showMessage(t.uploadFailed + (error.response?.data?.detail || error.message), 'error');
        progressSection.classList.add('hidden');
    } finally {
        document.getElementById('uploadBtn').disabled = false;
    }
}

async function refreshData() {
    try {
        await loadDatasets();
        updateStats();
    } catch (error) {
        console.error('Failed to refresh data:', error);
    }
}

async function loadDatasets() {
    try {
        const response = await axios.get(`${API_BASE}/api/datasets`);
        datasets = response.data.datasets || [];
        displayDatasets();
        updateStats();
    } catch (error) {
        console.error('Failed to load datasets:', error);
        showMessage((i18n[currentLang] || i18n.en).loadDatasetFailed, 'error');
    }
}

function displayDatasets() {
    const t = i18n[currentLang] || i18n.en;
    const container = document.getElementById('datasetsList');
    if (!container) return;
    if (datasets.length === 0) {
        container.innerHTML = `<p style="color: var(--muted);">${t.noDatasets}</p>`;
        return;
    }

    container.innerHTML = datasets.map(dataset => `
        <div class="file-item">
            <div>
                <strong>${formatDatasetDisplayName(dataset)}</strong>
                <span style="color: var(--muted);"> (${dataset.type})</span>
                ${dataset.type !== 'demo' ? `<span style="margin-left:8px; font-size:12px; color:${dataset.has_custom_schema ? '#1f8f55' : 'var(--muted)'}; font-weight:600;">${t.schemaTag}: ${dataset.has_custom_schema ? t.schemaCustom : t.schemaDefault}</span>` : ''}
            </div>
            <div class="dataset-actions">
                <span class="status-badge status-${dataset.status === 'ready' ? 'ready' : dataset.status === 'constructing' || dataset.status === 'reconstructing' ? 'processing' : 'error'}">
                    ${t.datasetStatusMap[dataset.status] || dataset.status}
                </span>
                ${dataset.status === 'needs_construction' ? 
                    `<button class="btn btn-primary" onclick="constructGraph('${dataset.name}')">${t.btnConstruct}</button>` : 
                    ''
                }
                ${dataset.status === 'ready' ? 
                    `<button class="btn btn-secondary" onclick="reconstructGraph('${dataset.name}')" title="${t.btnReconstruct}">${t.btnReconstruct}</button>` : 
                    ''
                }
                ${dataset.type !== 'demo' ? 
                    `<button class="btn" onclick="triggerSchemaUpload('${dataset.name}')" title="${t.btnUploadSchema}">${t.btnUploadSchema}</button>` : 
                    ''
                }
                ${dataset.type !== 'demo' ? 
                    `<button class="btn btn-danger" onclick="deleteDataset('${dataset.name}')" title="${t.btnDelete}">${t.btnDelete}</button>` : 
                    ''
                }
            </div>
        </div>
    `).join('');
}

// Ensure a hidden schema file input exists
function ensureSchemaFileInput() {
    let input = document.getElementById('schemaFileInput');
    if (!input) {
        input = document.createElement('input');
        input.type = 'file';
        input.id = 'schemaFileInput';
        input.accept = '.json';
        input.style.display = 'none';
        input.dataset.datasetName = '';
        input.addEventListener('change', async function (e) {
            const file = e.target.files && e.target.files[0];
            const datasetName = e.target.dataset.datasetName;
            if (!file || !datasetName) return;
            try {
                const t = i18n[currentLang] || i18n.en;
                showMessage(t.msgUploadingSchema(datasetName), 'info');
                const fd = new FormData();
                fd.append('schema_file', file);
                await axios.post(`${API_BASE}/api/datasets/${datasetName}/schema`, fd, {
                    headers: { 'Content-Type': 'multipart/form-data' }
                });
                showMessage(t.msgSchemaUploaded, 'success');
                await loadDatasets();
            } catch (err) {
                console.error('Schema upload failed:', err);
                const t = i18n[currentLang] || i18n.en;
                showMessage(t.msgSchemaUploadFailed + (err.response?.data?.detail || err.message), 'error');
            } finally {
                e.target.value = '';
                e.target.dataset.datasetName = '';
            }
        });
        document.body.appendChild(input);
    }
    return input;
}

function triggerSchemaUpload(datasetName) {
    const input = ensureSchemaFileInput();
    input.dataset.datasetName = datasetName;
    input.click();
}

async function constructGraph(datasetName) {
    // 前端立即将该数据集状态设为 constructing
    if (datasets && Array.isArray(datasets)) {
        for (let ds of datasets) {
            if (ds.name === datasetName) {
                ds.status = 'constructing';
            }
        }
        displayDatasets();
    }

    try {
        const t = i18n[currentLang] || i18n.en;
        showMessage(t.msgStartConstruct, 'info');

        // 建立WebSocket连接来接收实时进展
        const wsProto = window.location.protocol === 'https:' ? 'wss' : 'ws';
        const ws = new WebSocket(`${wsProto}://${window.location.host}/ws/web_client`);
        let progressMessages = [];

        ws.onopen = function() {
            console.log('WebSocket connected for progress updates');
        };

        ws.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'progress') {
                    progressMessages.push(data.message);
                    showMessage(t.msgConstructProgress(progressMessages.length, data.message), 'info');
                } else if (data.type === 'complete') {
                    showMessage(t.msgConstructCompleted, 'success');
                    // refresh datasets immediately to reflect ready status
                    refreshData();
                    ws.close();
                } else if (data.type === 'error') {
                    showMessage(t.msgConstructError(data.message), 'error');
                    refreshData();
                    ws.close();
                }
            } catch (e) {
                console.log('Progress update:', event.data);
            }
        };

        ws.onerror = function(error) {
            console.log('WebSocket error:', error);
        };

        ws.onclose = function() {
            console.log('WebSocket connection closed');
        };

        // Send construct request
        const response = await axios.post(`${API_BASE}/api/construct-graph`, {
            dataset_name: datasetName
        }, {
            params: { client_id: 'web_client' }
        });

        // If no WebSocket messages, close after timeout
        setTimeout(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.close();
                if (progressMessages.length === 0) {
                    showMessage(t.msgConstructCompleted, 'success');
                }
            }
        }, 30000); // 30s timeout

        // 不再立即 refreshData，等 WebSocket complete/error 后再刷新
    } catch (error) {
        console.error('Construction failed:', error);
        const t = i18n[currentLang] || i18n.en;
        showMessage(t.msgConstructFailed + (error.response?.data?.detail || error.message), 'error');
    }
}

async function reconstructGraph(datasetName) {
    const t = i18n[currentLang] || i18n.en;
    if (!confirm(t.confirmReconstruct(datasetName))) {
        return;
    }

    // 前端立即将该数据集状态设为 reconstructing
    if (datasets && Array.isArray(datasets)) {
        for (let ds of datasets) {
            if (ds.name === datasetName) {
                ds.status = 'reconstructing';
            }
        }
        displayDatasets();
    }

    try {
        showMessage(t.msgStartReconstruct, 'info');

        // Open WebSocket for real-time progress
        const wsProto = window.location.protocol === 'https:' ? 'wss' : 'ws';
        const ws = new WebSocket(`${wsProto}://${window.location.host}/ws/web_client`);
        let progressMessages = [];

        ws.onopen = function() {
            console.log('WebSocket connected for reconstruction progress updates');
        };

        ws.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'progress') {
                    progressMessages.push(data.message);
                    showMessage(t.msgReconstructProgress(progressMessages.length, data.message), 'info');
                } else if (data.type === 'complete') {
                    showMessage(t.msgReconstructCompleted, 'success');
                    // refresh datasets immediately to reflect ready status
                    refreshData();
                    ws.close();
                } else if (data.type === 'error') {
                    showMessage(t.msgReconstructError(data.message), 'error');
                    refreshData();
                    ws.close();
                }
            } catch (e) {
                console.log('Reconstruction progress update:', event.data);
            }
        };

        ws.onerror = function(error) {
            console.log('WebSocket error:', error);
        };

        ws.onclose = function() {
            console.log('WebSocket connection closed');
        };

        const response = await axios.post(`${API_BASE}/api/datasets/${datasetName}/reconstruct`, {}, {
            params: { client_id: 'web_client' }
        });

        // If no WebSocket messages, close after timeout
        setTimeout(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.close();
                if (progressMessages.length === 0) {
                    showMessage(t.msgReconstructCompleted, 'success');
                }
            }
        }, 30000); // 30s timeout

        // 不再立即 refreshData，等 WebSocket complete/error 后再刷新
    } catch (error) {
        console.error('Reconstruction failed:', error);
        showMessage(t.msgReconstructFailed + (error.response?.data?.detail || error.message), 'error');
    }
}

async function deleteDataset(datasetName) {
    const t = i18n[currentLang] || i18n.en;
    if (!confirm(t.confirmDelete(datasetName))) {
        return;
    }

    try {
        showMessage(t.msgDeletingDataset, 'info');
        const response = await axios.delete(`${API_BASE}/api/datasets/${datasetName}`);
        showMessage(t.msgDatasetDeleted(datasetName), 'success');
        refreshData();

        // If the deleted dataset was selected, clear selections
        const graphDataset = document.getElementById('graphDataset');
        const qaDataset = document.getElementById('qaDataset');
        if (graphDataset.value === datasetName) {
            graphDataset.value = '';
        }
        if (qaDataset.value === datasetName) {
            qaDataset.value = '';
        }
    } catch (error) {
        console.error('Deletion failed:', error);
        showMessage(t.msgDeleteFailed + (error.response?.data?.detail || error.message), 'error');
    }
}

function updateStats() {
    document.getElementById('stat-datasets').textContent = datasets.length;
    document.getElementById('stat-graphs').textContent = datasets.filter(d => d.status === 'ready').length;
    document.getElementById('stat-questions').textContent = questionCount;
}

function loadDatasetOptions(selectId) {
    const select = document.getElementById(selectId);
    if (!select) return;

    const readyDatasets = datasets.filter(d => d.status === 'ready');
    const currentValue = select.value;
    const langConfig = i18n[currentLang] || i18n.en;
    const placeholder = langConfig?.selectDatasetPlaceholder || 'Select a dataset...';

    select.innerHTML = `<option value="">${placeholder}</option>` +
        readyDatasets.map(d => `<option value="${d.name}">${formatDatasetDisplayName(d)}</option>`).join('');

    if (readyDatasets.some(d => d.name === currentValue)) {
        select.value = currentValue;
    }
}

async function loadGraphData() {
    const datasetName = document.getElementById('graphDataset').value;
    if (!datasetName) return;

    try {
        const response = await axios.get(`${API_BASE}/api/graph/${datasetName}`);
        const graphData = response.data;
        renderGraph(graphData);
    } catch (error) {
        console.error('Failed to load graph data:', error);
        showMessage((i18n[currentLang] || i18n.en).msgLoadGraphFailed, 'error');
    }
}

function renderGraph(data) {
    const chartContainer = document.getElementById('graphChart');
    const t = i18n[currentLang] || i18n.en;

    if (graphChart) {
        graphChart.dispose();
    }

    graphChart = echarts.init(chartContainer);

    const option = {
        backgroundColor: '#13233a',
        // title: {
        //     text: 'Knowledge Graph',
        //     left: 'center',
        //     textStyle: { color: 'rgba(255, 255, 255, 0.95)' }
        // },
        tooltip: {
            trigger: 'item',
            backgroundColor: 'rgba(10, 18, 30, 0.92)',
            borderColor: 'rgba(125, 170, 228, 0.45)',
            borderWidth: 1,
            textStyle: { color: '#f4f8ff' },
            formatter: function (params) {
                if (params.dataType === 'node') {
                    const d = params.data || {};
                    let name = (d.fullName || d.name || '').toString().replace(/\s+/g,' ').trim();
                    if (name.length > 40) name = name.slice(0,40) + '...';
                    const category = d.category || d.type || '';
                    return `${name || 'node'}<br/>${t.nodeTypeLabel}: ${category || 'entity'}`;
                } else if (params.dataType === 'edge') {
                    const rel = params.data && params.data.name ? params.data.name : '';
                    return `${t.relationLabel}: ${rel}`;
                }
                return '';
            }
        },
        legend: {
            type: 'scroll',
            bottom: 10,
            textStyle: { color: '#dce9ff' },
            data: (data.categories && data.categories.map(function(c){return c.name;})) || []
        },
        series: [{
            type: 'graph',
            layout: 'force',
            data: data.nodes || [],
            links: data.links || [],
            categories: data.categories || [],
            roam: true,
            label: {
                show: (data.nodes || []).length <= 500,
                color: '#e8f3ff',
                textShadowColor: 'rgba(0,0,0,0.55)',
                textShadowBlur: 4,
                formatter: function(p){
                    const d = p.data || {};
                    let name = (d.fullName || d.name || '').toString().replace(/\s+/g,' ').trim();
                    if (name.length > 20) name = name.slice(0,20) + '...';
                    return name || '';
                }
            },
            force: {
                repulsion: 1000,
                gravity: 0.1,
                edgeLength: 120
            },
            lineStyle: {
                opacity: 0.85,
                width: 1.6,
                color: 'rgba(144, 185, 236, 0.78)'
            }
        }]
    };

    graphChart.setOption(option);
}

function renderKnowledgeGraphChart(data) {
    const chartContainer = document.getElementById('triplesChart');
    const t = i18n[currentLang] || i18n.en;
    if (!chartContainer) return;

    if (qaSubgraphChart) {
        qaSubgraphChart.dispose();
    }
    qaSubgraphChart = echarts.init(chartContainer);

    const option = {
        backgroundColor: '#13233a',
        tooltip: {
            trigger: 'item',
            backgroundColor: 'rgba(10, 18, 30, 0.92)',
            borderColor: 'rgba(125, 170, 228, 0.45)',
            borderWidth: 1,
            textStyle: { color: '#f4f8ff' },
            formatter: function (params) {
                if (params.dataType === 'node') {
                    const d = params.data || {};
                    let name = (d.name || '').toString().replace(/\s+/g, ' ').trim();
                    if (name.length > 20) name = name.slice(0, 20) + '...';
                    const category = d.category || d.type || '';
                    const description = (d.description || '').toString().replace(/\s+/g, ' ').trim();
                    const parts = [`${name || 'node'}`, `${t.nodeTypeLabel}: ${category || 'entity'}`];
                    if (description) {
                        const shortDescription = description.length > 120 ? `${description.slice(0, 120)}...` : description;
                        parts.push(`${t.descriptionLabel || 'Description'}: ${shortDescription}`);
                    }
                    return parts.join('<br/>');
                } else if (params.dataType === 'edge') {
                    const rel = params.data && params.data.name ? params.data.name : '';
                    return `${t.relationLabel}: ${rel}`;
                }
                return '';
            }
        },
        legend: {
            type: 'scroll',
            bottom: 10,
            textStyle: { color: '#dce9ff' },
            data: (data.categories && data.categories.map(function (c) { return c.name; })) || []
        },
        series: [{
            type: 'graph',
            layout: 'force',
            data: data.nodes || [],
            links: data.links || [],
            categories: data.categories || [],
            roam: true,
            label: {
                show: true,
                color: '#e8f3ff',
                textShadowColor: 'rgba(0,0,0,0.55)',
                textShadowBlur: 4,
                formatter: function (p) {
                    const d = p.data || {};
                    let name = (d.name || '').toString().replace(/\s+/g, ' ').trim();
                    if (name.length > 20) name = name.slice(0, 20) + '...';
                    return name || '';
                }
            },
            force: {
                repulsion: 1000,
                gravity: 0.1,
                edgeLength: 120
            },
            lineStyle: {
                opacity: 0.85,
                width: 1.6,
                color: 'rgba(144, 185, 236, 0.78)'
            }
        }]
    };

    qaSubgraphChart.setOption(option);
    setTimeout(() => {
        if (qaSubgraphChart && typeof qaSubgraphChart.resize === 'function') {
            qaSubgraphChart.resize();
        }
    }, 100);
}

async function askQuestion() {
    const datasetName = document.getElementById('qaDataset').value;
    const question = document.getElementById('questionInput').value.trim();

    if (!datasetName || !question) {
        showMessage(i18n[currentLang].pleaseSelectAndEnter, 'error');
        return;
    }

    // Clear previous QA inline messages when starting a new question
    const qaMsgs = document.getElementById('qaMessages');
    if (qaMsgs) qaMsgs.innerHTML = '';

    const loading = document.getElementById('qaLoading');
    const answerSection = document.getElementById('answerSection');
    const askBtn = document.getElementById('askBtn');
    const qaBadge = document.getElementById('qaCompletionBadge');

    loading.classList.add('show');
    const procText = document.getElementById('qaProcessingText');
    if (procText) procText.textContent = i18n[currentLang].processing;
    answerSection.classList.add('hidden');
    document.getElementById('answerContent').textContent = '';
    askBtn.disabled = true;
    if (qaBadge) qaBadge.style.display = 'none';

    let ws = null;
    let qaCompleted = false;
    try {
        // Open WebSocket for live QA updates
        const wsProto = window.location.protocol === 'https:' ? 'wss' : 'ws';
        ws = new WebSocket(`${wsProto}://${window.location.host}/ws/web_client`);
        let wsOpen = false;
        ws.onopen = () => { wsOpen = true; };
        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'qa_update') {
                    if (data.stage === 'decompose' && data.decompose_fallback) {
                        showMessage((i18n[currentLang] || i18n.en).decomposeFallbackNotice, 'warning', 10000);
                    } else if (data.stage === 'sub_question') {
                        const q = (data.question || '').toString();
                        if (currentLang === 'zh') {
                            const msg = `【子问题 ${data.index}/${data.total}】${q}｜三元组：${data.triples_count || 0}｜片段：${data.chunks_count || 0}`;
                            showMessage(msg, 'info');
                            if (Array.isArray(data.triples_preview) && data.triples_preview.length) {
                                showMessage((i18n[currentLang] || i18n.en).previewPrefix + data.triples_preview.slice(0,3).join(' | '), 'info');
                            }
                        } else {
                            const msg = `[Sub ${data.index}/${data.total}] ${q} | Triples: ${data.triples_count || 0} | Chunks: ${data.chunks_count || 0}`;
                            showMessage(msg, 'info');
                            if (Array.isArray(data.triples_preview) && data.triples_preview.length) {
                                showMessage((i18n[currentLang] || i18n.en).previewPrefix + data.triples_preview.slice(0,3).join(' | '), 'info');
                            }
                        }
                    } else if (data.stage === 'ircot') {
                        const thought = (data.thought_preview || '').toString().slice(0,140) + '...';
                        const msg = currentLang === 'zh' ? `IRCoT 步骤 ${data.step}/${data.max_steps}：${thought}` : `IRCoT Step ${data.step}/${data.max_steps}: ${thought}`;
                        showMessage(msg, 'info');
                    }
                } else if (data.type === 'qa_complete') {
                    qaCompleted = true;
                    showMessage(i18n[currentLang].qaCompleted, 'success-bright', 12000);
                    const badge = document.getElementById('qaCompletionBadge');
                    if (badge) { badge.style.display = 'inline-flex'; badge.textContent = i18n[currentLang].completedBadge; }
                    // Do not close here; HTTP response will render the final details
                    setTimeout(() => { try { if (ws && ws.readyState === WebSocket.OPEN) ws.close(); } catch(_){} }, 1000);
                }
            } catch (e) {
                // non-JSON message
            }
        };
        ws.onerror = () => {};

        const response = await axios.post(`${API_BASE}/api/ask-question`, {
            question: question,
            dataset_name: datasetName
        }, {
            params: { client_id: 'web_client' }
        });

        const result = response.data;
        displayAnswer(result);
        questionCount++;
        updateStats();

    } catch (error) {
        console.error('Question failed:', error);
        showMessage((i18n[currentLang] || i18n.en).msgQuestionFailed + (error.response?.data?.detail || error.message), 'error');
    } finally {
        // Delay hiding the loader briefly when completed so the bright message is visible
        const hide = () => loading.classList.remove('show');
        if (qaCompleted) {
            setTimeout(hide, 1500);
        } else {
            hide();
        }
        askBtn.disabled = false;
        try { if (ws && ws.readyState === WebSocket.OPEN) ws.close(); } catch(_){}
    }
}

// displayAnswer 的主版本在后面定义，此处已移除重复版本

/**
 * 点击引用 [N] 后，平滑滚动到参考文献区的对应条目并高亮
 */
function scrollToReference(index) {
    const targetId = 'ref-' + index;
    const targetEl = document.getElementById(targetId);
    if (!targetEl) {
        // 如果目标不存在（该篇论文未在引用中），给用户友好提示
        const refSection = document.querySelector('.reference-section');
        if (refSection) {
            refSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        return;
    }

    // 平滑滚动到目标元素
    targetEl.scrollIntoView({ behavior: 'smooth', block: 'center' });

    // 高亮效果：先移除已有高亮，再添加新的
    document.querySelectorAll('.reference-item.ref-highlight').forEach(el => {
        el.classList.remove('ref-highlight');
    });
    targetEl.classList.add('ref-highlight');

    // 2秒后自动移除高亮
    setTimeout(() => {
        targetEl.classList.remove('ref-highlight');
    }, 2000);
}

function escapeHtml(text) {
    return String(text || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function renderInlineMarkdown(escapedText) {
    let html = String(escapedText || '');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    // 引用链接：使用 onclick 平滑滚动，而非原生锚点（SPA 页面中原生锚点可能因嵌套滚动而失效）
    html = html.replace(/\[(\d+)\]/g, '<a href="javascript:void(0)" class="citation" onclick="scrollToReference($1)">[$1]</a>');
    return html;
}

function renderSafeMarkdown(markdownText) {
    const lines = String(markdownText || '').replace(/\r\n/g, '\n').split('\n');
    const output = [];
    let inCode = false;
    let inUl = false;
    let inOl = false;
    let paragraph = [];

    const flushParagraph = () => {
        if (!paragraph.length) return;
        output.push(`<p>${renderInlineMarkdown(paragraph.join('<br/>'))}</p>`);
        paragraph = [];
    };

    const closeLists = () => {
        if (inUl) {
            output.push('</ul>');
            inUl = false;
        }
        if (inOl) {
            output.push('</ol>');
            inOl = false;
        }
    };

    for (const rawLine of lines) {
        const line = escapeHtml(rawLine);
        const trimmed = line.trim();

        if (trimmed.startsWith('```')) {
            flushParagraph();
            closeLists();
            if (!inCode) {
                inCode = true;
                output.push('<pre><code>');
            } else {
                inCode = false;
                output.push('</code></pre>');
            }
            continue;
        }

        if (inCode) {
            output.push(line);
            continue;
        }

        if (!trimmed) {
            flushParagraph();
            closeLists();
            continue;
        }

        const heading = trimmed.match(/^#{1,3}\s+(.+)$/);
        if (heading) {
            flushParagraph();
            closeLists();
            output.push(`<h4>${renderInlineMarkdown(heading[1])}</h4>`);
            continue;
        }

        const ulItem = trimmed.match(/^[-*]\s+(.+)$/);
        if (ulItem) {
            flushParagraph();
            if (inOl) {
                output.push('</ol>');
                inOl = false;
            }
            if (!inUl) {
                output.push('<ul>');
                inUl = true;
            }
            output.push(`<li>${renderInlineMarkdown(ulItem[1])}</li>`);
            continue;
        }

        const olItem = trimmed.match(/^\d+\.\s+(.+)$/);
        if (olItem) {
            flushParagraph();
            if (inUl) {
                output.push('</ul>');
                inUl = false;
            }
            if (!inOl) {
                output.push('<ol>');
                inOl = true;
            }
            output.push(`<li>${renderInlineMarkdown(olItem[1])}</li>`);
            continue;
        }

        closeLists();
        paragraph.push(trimmed);
    }

    flushParagraph();
    closeLists();
    if (inCode) output.push('</code></pre>');
    return output.join('\n');
}

function stripMarkdown(text) {
    return String(text || '')
        .replace(/```[\s\S]*?```/g, ' ')
        .replace(/`([^`]+)`/g, '$1')
        .replace(/\*\*([^*]+)\*\*/g, '$1')
        .replace(/\*([^*]+)\*/g, '$1')
        .replace(/#{1,6}\s+/g, '')
        .replace(/>\s*/g, '')
        .replace(/\s+/g, ' ')
        .trim();
}

function normalizeAnswerText(text) {
    return String(text || '')
        .replace(/\r\n/g, '\n')
        .replace(/\n{3,}/g, '\n\n')
        .replace(/[ \t]+\n/g, '\n')
        .trim();
}

function cleanAnswerArtifacts(text) {
    let cleaned = normalizeAnswerText(text);
    const soAnswerMatch = cleaned.match(/So the answer is:\s*([\s\S]*)$/i);
    if (soAnswerMatch && soAnswerMatch[1]) {
        cleaned = soAnswerMatch[1].trim();
    }
    cleaned = cleaned
        .replace(/\*\*推理过程\*\*[:：]?/g, '')
        .replace(/\*\*结论\*\*[:：]?/g, '')
        .replace(/推理过程[:：]/g, '')
        .replace(/结论[:：]/g, '')
        .replace(/^根据提供的(?:知识|检索)?上下文(?:，|,)?(?:可以明确回答(?:当前)?问题。?)?/g, '')
        .replace(/^可以明确回答(?:当前)?问题。?/g, '')
        .trim();
    return cleaned;
}

function extractFirstSentence(text) {
    const cleaned = stripMarkdown(cleanAnswerArtifacts(text));
    if (!cleaned) return '';
    const parts = cleaned.split(/(?<=[。！？!?])/).map((part) => part.trim()).filter(Boolean);
    return parts.length ? parts[0] : cleaned;
}

// Override the earlier answer renderer: keep a single compact answer block.
function displayAnswer(result) {
    console.log('displayAnswer called with result:', result);
    const answerContent = document.getElementById('answerContent');
    const answerSection = document.getElementById('answerSection');
    
    // Remove old references wrapper if any
    const oldRefs = document.getElementById('referencesContainerWrapper');
    if (oldRefs) oldRefs.remove();

    let processedAnswer = result.answer || '';
    const papers = result.retrieved_papers || [];
    let citedPapers = [];
    
    if (Array.isArray(papers) && papers.length > 0) {
        const citedMatches = Array.from(processedAnswer.matchAll(/\[(\d+)\]/g));
        const indexMap = new Map();
        let nextIndex = 1;
        
        citedMatches.forEach(m => {
            const oldIdx = parseInt(m[1], 10);
            if (!indexMap.has(oldIdx)) {
                indexMap.set(oldIdx, nextIndex++);
                const paper = papers.find(p => p.index === oldIdx);
                if (paper) {
                    citedPapers.push({...paper, index: indexMap.get(oldIdx)});
                }
            }
        });
        
        processedAnswer = processedAnswer.replace(/\[(\d+)\]/g, (match, num) => {
            const oldIdx = parseInt(num, 10);
            return indexMap.has(oldIdx) ? `[${indexMap.get(oldIdx)}]` : match;
        });
    }

    if (answerContent) {
        answerContent.className = 'answer-content answer-layout';
        answerContent.innerHTML = `
            <div class="answer-lead">
                <div class="answer-section-title">${currentLang === 'zh' ? '综合回答' : 'Comprehensive Answer'}</div>
                <div class="answer-lead-text">${renderSafeMarkdown(processedAnswer) || '<p></p>'}</div>
            </div>
        `;
    }
    
    if (result.decompose_fallback) {
        showMessage((i18n[currentLang] || i18n.en).decomposeFallbackNotice, 'warning', 10000);
    }

    displayRetrievalDetails(result);
    answerSection.classList.remove('hidden');

    if (citedPapers.length > 0) {
        const citedSet = new Set(citedPapers.map(p => p.index));
        const refsHtml = renderReferences(citedPapers, citedSet);
        if (refsHtml) {
            const refWrapper = document.createElement('div');
            refWrapper.id = 'referencesContainerWrapper';
            refWrapper.innerHTML = refsHtml;
            answerSection.appendChild(refWrapper);
        }
    }
}

function filterKnowledgeGraphForQA(kg) {
    if (!kg || !Array.isArray(kg.nodes)) {
        return { nodes: [], links: [], categories: [] };
    }

    const links = Array.isArray(kg.links) ? kg.links : [];
    const rawNodes = Array.isArray(kg.nodes) ? kg.nodes : [];
    const degreeMap = new Map();
    const componentSizeMap = new Map();

    rawNodes.forEach((node) => {
        const nodeId = String(node?.id ?? '');
        if (!nodeId) return;
        const degree = Number(node?.degree ?? 0);
        degreeMap.set(nodeId, degree);
        const componentId = Number(node?.component_id ?? -1);
        componentSizeMap.set(componentId, (componentSizeMap.get(componentId) || 0) + 1);
    });

    if (!degreeMap.size) {
        links.forEach((link) => {
            const source = String(typeof link.source === 'object' ? link.source.id : link.source);
            const target = String(typeof link.target === 'object' ? link.target.id : link.target);
            degreeMap.set(source, (degreeMap.get(source) || 0) + 1);
            degreeMap.set(target, (degreeMap.get(target) || 0) + 1);
        });
    }

    const dominantComponentId = Array.from(componentSizeMap.entries())
        .sort((a, b) => b[1] - a[1])[0]?.[0];

    const keptNodes = rawNodes.filter((node) => {
        const category = String((node && (node.category || node.type)) || '').trim().toLowerCase();
        const nodeId = String(node?.id ?? '');
        const degree = Number(node?.degree ?? degreeMap.get(nodeId) ?? 0);
        const componentId = Number(node?.component_id ?? -1);
        const isFallbackEntity = Boolean(node?.is_fallback_entity);

        // Strategy change: Simplified filtration to preserve the "continuous" feel of the graph.
        // We now trust the backend's selection of nodes and only filter if they are completely orphaned entities.
        if (category !== 'entity') {
            return true;
        }

        // For generic entities, keep them if they represent at least one piece of evidence (degree >= 1).
        if (degree >= 1) {
            return true;
        }
        
        // If it's a fallback entity from unstructured extraction, still prefer showing it if it's connected.
        return !isFallbackEntity;
    });

    const nodeIdSet = new Set(keptNodes.map((node) => String(node.id)));
    const keptLinks = links.filter((link) => {
        const source = typeof link.source === 'object' ? link.source.id : link.source;
        const target = typeof link.target === 'object' ? link.target.id : link.target;
        return nodeIdSet.has(String(source)) && nodeIdSet.has(String(target));
    });

    const categorySet = new Set(
        keptNodes.map((node) => String((node && (node.category || node.type)) || 'entity'))
    );

    let categories = (kg.categories || []).filter((cat) => {
        const name = String((cat && cat.name) || '').trim();
        return name && categorySet.has(name);
    });
    if (!categories.length) {
        categories = Array.from(categorySet).map(name => ({ name }));
    }

    return { nodes: keptNodes, links: keptLinks, categories };
}


function displayRetrievalDetails(result) {
    console.log('displayRetrievalDetails called with:', result);
    const t = i18n[currentLang] || i18n.en;

    // Helper: deduplicate triples array of strings '(s, r, o)' retaining order
    function dedupTriples(arr) {
        if (!Array.isArray(arr)) return [];
        const seen = new Set();
        const out = [];
        for (const t of arr) {
            if (typeof t !== 'string') continue;
            const m = t.match(/\(([^,]+),\s*([^,]+),\s*([^\)]+)\)/);
            if (!m) continue;
            const key = m.slice(1).map(x => x.trim().toLowerCase()).join('|');
            if (!seen.has(key)) {
                seen.add(key);
                out.push(`(${m[1].trim()}, ${m[2].trim()}, ${m[3].trim()})`);
            }
        }
        return out;
    }

    function dedupChunks(arr) {
        if (!Array.isArray(arr)) return [];
        const seen = new Set();
        const out = [];
        for (const item of arr) {
            if (typeof item !== 'string') continue;
            const normalized = item.replace(/\s+/g, ' ').trim();
            if (!normalized) continue;
            const key = normalized.toLowerCase();
            if (!seen.has(key)) {
                seen.add(key);
                out.push(normalized);
            }
        }
        return out;
    }

    const detailsContainer = document.getElementById('retrievalDetails');
    if (!detailsContainer) {
        console.error('Retrieval details container not found');
        return;
    }

    console.log('Found retrieval details container, showing it');

    // Show the container
    detailsContainer.style.display = 'block';

    // Compute total retrieved triples from visible sub-question steps (sum of each step's deduped triples list)
    let subQuestionTriplesTotal = 0;
    if (result.sub_questions && result.sub_questions.length > 0 && Array.isArray(result.reasoning_steps)) {
        for (let i = 0; i < result.sub_questions.length; i++) {
            const step = result.reasoning_steps[i];
            if (step && Array.isArray(step.triples)) {
                subQuestionTriplesTotal += dedupTriples(step.triples).length;
            } else if (step && typeof step.triples_count === 'number') {
                // fallback if only count exists
                subQuestionTriplesTotal += step.triples_count;
            }
        }
    }

    let detailsHtml = `
        <div class="retrieval-summary">
    <h4>${t.retrievalStatsTitle}</h4>
            <div class="stats-grid">
                <div class="stat-item">
        <span class="stat-label">${t.retrievalSubQuestions}</span>
                    <span class="stat-value">${result.sub_questions?.length || 0}</span>
                </div>
                <div class="stat-item">
        <span class="stat-label">${t.retrievalTriplesTotal}</span>
                    <span class="stat-value">${subQuestionTriplesTotal}</span>
                </div>
                <div class="stat-item">
        <span class="stat-label">${t.retrievalChunks}</span>
                    <span class="stat-value">${result.retrieved_chunks?.length || 0}</span>
                </div>
            </div>
        </div>
        
        <div class="subquestions-section">
    <h4>${t.questionDecompositionTitle}</h4>
            <div class="subquestions-list">
    `;

    if (result.sub_questions && result.sub_questions.length > 0) {
        result.sub_questions.forEach((sq, index) => {
            const step = result.reasoning_steps?.[index] || {};
            const stepTriples = dedupTriples(step.triples || []);
            const stepChunks = dedupChunks(step.chunk_contents || result.retrieved_chunks || []);

            // 1. 生成新的 HTML 逻辑
            const visibleTriples = stepTriples.slice(0, 2);
            const hiddenTriples = stepTriples.slice(2);
            const visibleChunks = stepChunks.slice(0, 2);
            const hiddenChunks = stepChunks.slice(2);

            let triplesHtml = '';
            if (stepTriples.length > 0) {
                triplesHtml = `
                    <div class="triples-preview">
                        <strong>${t.retrievedTriplesLabel}</strong>
                        <ul>
                            ${visibleTriples.map(triple => `<li>${triple}</li>`).join('')}
                        </ul>
                        
                        ${hiddenTriples.length > 0 ? `
                            <ul id="hidden-triples-${index}" style="display: none; margin-top: 0;">
                                ${hiddenTriples.map(triple => `<li>${triple}</li>`).join('')}
                            </ul>
                            <span class="more-indicator" 
                                  style="cursor: pointer; text-decoration: underline; color: #3b82f6;" 
                                  onclick="toggleHiddenList('hidden-triples-${index}', this, 'triples')">
                                ${t.moreTriples(hiddenTriples.length)}
                            </span>
                        ` : ''}
                    </div>
                `;
            }

            let chunksHtml = '';
            if (stepChunks.length > 0) {
                chunksHtml = `
                    <div class="triples-preview">
                        <strong>${t.retrievedChunksLabel}</strong>
                        <ul>
                            ${visibleChunks.map(chunk => `<li>${chunk}</li>`).join('')}
                        </ul>

                        ${hiddenChunks.length > 0 ? `
                            <ul id="hidden-chunks-${index}" style="display: none; margin-top: 0;">
                                ${hiddenChunks.map(chunk => `<li>${chunk}</li>`).join('')}
                            </ul>
                            <span class="more-indicator"
                                  style="cursor: pointer; text-decoration: underline; color: #3b82f6;"
                                  onclick="toggleHiddenList('hidden-chunks-${index}', this, 'chunks')">
                                ${t.moreTriples(hiddenChunks.length)}
                            </span>
                        ` : ''}
                    </div>
                `;
            }

            // 2. 将生成的 triplesHtml 拼接到主字符串中
            // 注意：这里删除了你原代码中重复的旧逻辑，直接插入 ${triplesHtml}
            detailsHtml += `
                <div class="subquestion-item">
                    <div class="subquestion-header">
                        <span class="subquestion-number">${index + 1}</span>
                        <span class="subquestion-text">${sq['sub-question'] || sq.question}</span>
                    </div>
                    <div class="subquestion-stats">
                        <span>${t.triplesLabel}: ${stepTriples.length}</span>
                        <span>${t.chunksLabel}: ${stepChunks.length || step.chunks_count || 0}</span>
                        <span>${t.timeLabel}: ${(step.processing_time || 0).toFixed(2)}s</span>
                    </div>
                    ${triplesHtml}
                    ${chunksHtml}
                </div>
            `;
        });
    } else {
        detailsHtml += `<p class="no-data">${t.noSubQuestionData}</p>`;
    }

    detailsHtml += `
            </div>
        </div>
        
        <div class="triples-section">
            <h4>${t.subgraphTitle}</h4>
            <div class="triples-chart-container">
                <div id="triplesChart" class="chart-container" style="height: 400px;"></div>
            </div>
            <div class="triples-summary">
                <p>${t.retrievedTriplesSummary(result.retrieved_triples?.length || 0)}</p>
            </div>
        </div>
        
        ${/* references rendered separately after innerHTML set */''}  
    `;

    detailsHtml += '</div></div>';
    detailsContainer.innerHTML = detailsHtml;

    // References are now rendered dynamically alongside answer in displayAnswer

    // Render retrieval subgraph after setting HTML and ensuring container is visible
    setTimeout(() => {
        console.log('About to render triples chart with:', result.retrieved_triples?.length || 0, 'triples');
        // Make sure the chart container is visible before initializing ECharts
        const chartContainer = document.getElementById('triplesChart');
        if (chartContainer) {
            chartContainer.style.display = 'block';
            console.log('Chart container made visible');
        }
        const kg = result?.visualization_data?.knowledge_graph;
        const filteredKg = filterKnowledgeGraphForQA(kg);
        if (filteredKg && Array.isArray(filteredKg.nodes) && filteredKg.nodes.length > 0) {
            renderKnowledgeGraphChart(filteredKg);
        } else if (kg && Array.isArray(kg.nodes) && chartContainer) {
            chartContainer.innerHTML = `<div style="color: #ffb347; text-align: center; padding: 50px;">${
                currentLang === 'zh'
                    ? '\u672a\u53d1\u73b0\u53ef\u5c55\u793a\u7684\u4e3b\u8981\u63a8\u7406\u94fe'
                    : 'No primary reasoning chain available'
            }</div>`;
        } else {
            renderTriplesChart(result.retrieved_triples || [], { hideEntity: false });
        }
    }, 200);
}

function renderTriplesChart(triples, options = {}) {
    console.log('renderTriplesChart called with:', triples.length, 'triples');
    console.log('First triple example:', triples[0]);
    const t = i18n[currentLang] || i18n.en;
    const hideEntity = Boolean(options.hideEntity);

    // Helper: truncate by words. If more than 3 words, keep first 2 then ellipsis.
    function truncateWords(str){
        if(!str) return '';
        const words = str.split(/\s+/).filter(Boolean);
        if(words.length <= 3) return words.join(' ');
        return words.slice(0,2).join(' ') + '...';
    }

    const chartContainer = document.getElementById('triplesChart');
    if (!chartContainer) {
        console.error('Triples chart container not found');
        return;
    }

    console.log('Found chart container, container dimensions:', chartContainer.offsetWidth, 'x', chartContainer.offsetHeight);

    // Check if ECharts is available
    if (typeof echarts === 'undefined') {
        console.error('ECharts library is not loaded!');
        chartContainer.innerHTML = `<div style="color: red; text-align: center; padding: 50px;">${t.echartsNotLoaded}</div>`;
        return;
    }

    console.log('ECharts library available, version:', echarts.version);

    // Dispose existing chart
    if (window.triplesChart && typeof window.triplesChart.dispose === 'function') {
        try {
            window.triplesChart.dispose();
            console.log('Previous chart disposed');
        } catch (e) {
            console.warn('Error disposing previous chart:', e);
        }
    }
    window.triplesChart = null;

    try {
        window.triplesChart = echarts.init(chartContainer);
        console.log('ECharts initialized successfully');
    } catch (e) {
        console.error('Error initializing ECharts:', e);
        chartContainer.innerHTML = `<div style="color: red; text-align: center; padding: 50px;">${t.echartsInitError}${e.message}</div>`;
        return;
    }

    // Parse triples and build graph data
    const nodes = new Map();
    const links = [];
    const categories = new Set();

    console.log('Parsing', triples.length, 'triples...');

    triples.forEach((tripleStr, index) => {
        // Parse triple format: (subject, relation, object) [score: X.XXX]
        const match = tripleStr.match(/\(([^,]+),\s*([^,]+),\s*([^)]+)\)/);
        if (match) {
            if (index < 3) {
                console.log(`Triple ${index + 1} parsed successfully:`, match.slice(1));
            }
            let [, subject, relation, object] = match;

            // Clean up entity names and extract schema types
            subject = subject.trim();
            relation = relation.trim();
            object = object.trim();

            // Extract schema type if present
            const subjectType = extractSchemaType(subject) || 'entity';
            const objectType = extractSchemaType(object) || 'entity';

            // Clean entity names
            subject = cleanEntityName(subject);
            object = cleanEntityName(object);

            // Optional filtering for QA subgraph: hide generic fallback type
            if (hideEntity && (subjectType.toLowerCase() === 'entity' || objectType.toLowerCase() === 'entity')) {
                return;
            }

            // Add nodes
            if (!nodes.has(subject)) {
                nodes.set(subject, {
                    id: subject,
                    name: truncateWords(subject),
                    category: subjectType,
                    symbolSize: Math.min(30 + subject.length * 0.5, 50),
                    itemStyle: { color: getCategoryColor(subjectType) },
                    rawName: subject
                });
                categories.add(subjectType);
            }

            if (!nodes.has(object)) {
                nodes.set(object, {
                    id: object,
                    name: truncateWords(object),
                    category: objectType,
                    symbolSize: Math.min(30 + object.length * 0.5, 50),
                    itemStyle: { color: getCategoryColor(objectType) },
                    rawName: object
                });
                categories.add(objectType);
            }

            // Add link with truncated display name and rawName preserved
            links.push({
                source: subject,
                target: object,
                name: truncateWords(relation),
                rawName: relation,
                lineStyle: {
                    color: '#8fc2ff',
                    width: 2.2,
                    opacity: 0.88
                }
            });
        }
    });

    const nodesList = Array.from(nodes.values());
    const categoriesList = Array.from(categories).map(cat => ({
        name: cat,
        itemStyle: { color: getCategoryColor(cat) }
    }));

    const option = {
        backgroundColor: '#13233a',
        // title: {
        //     text: '检索三元组知识图谱',
        //     left: 'center',
        //     top: 10,
        //     textStyle: {
        //         color: 'rgba(255, 255, 255, 0.95)',
        //         fontSize: 16
        //     }
        // },
        tooltip: {
            trigger: 'item',
            backgroundColor: 'rgba(10, 18, 30, 0.92)',
            borderColor: 'rgba(125, 170, 228, 0.45)',
            borderWidth: 1,
            textStyle: { color: '#f4f8ff' },
            formatter: function(params) {
                if (params.dataType === 'node') {
                    const rawFull = (params.data.rawName || params.data.id || '').toString().replace(/\s+/g,' ').trim();
                    return `<strong>${rawFull}</strong><br/>${t.nodeTypeLabel}: ${params.data.category}`;
                } else if (params.dataType === 'edge') {
                    const fullRel = (params.data.rawName || params.data.name || '').toString().replace(/\s+/g,' ').trim();
                    return `<strong>${fullRel}</strong><br/>${params.data.source} → ${params.data.target}`;
                }
                return '';
            }
        },
        legend: {
            type: 'scroll',
            bottom: 10,
            textStyle: { color: '#dce9ff' },
            data: categoriesList.map(c => c.name)
        },
        series: [{
            type: 'graph',
            layout: 'force',
            data: nodesList,
            links: links,
            categories: categoriesList,
            roam: true,
            focusNodeAdjacency: true,
            label: {
                show: true,
                position: 'right',
                color: '#e8f3ff',
                textShadowColor: 'rgba(0,0,0,0.55)',
                textShadowBlur: 4,
                fontSize: 10,
                formatter: function(p){
                    let name = (p.data && (p.data.rawName || p.data.name || p.data.id) || '').toString().replace(/\s+/g,' ').trim();
                    return truncateWords(name);
                }
            },
            force: {
                repulsion: 1000,
                gravity: 0.1,
                edgeLength: [100, 200],
                layoutAnimation: true
            },
            lineStyle: {
                opacity: 0.9,
                curveness: 0.1
            },
            edgeLabel: {
                show: true,
                color: '#dbe9ff',
                fontSize: 10,
                formatter: function(p){
                    return truncateWords(p.data && (p.data.rawName || p.data.name) || '');
                }
            },
            emphasis: {
                focus: 'adjacency',
                lineStyle: {
                    width: 4
                }
            }
        }]
    };

    console.log('Setting chart option with:', nodesList.length, 'nodes and', links.length, 'links');

    if (nodesList.length === 0) {
        console.warn('No nodes to display, showing empty message');
        chartContainer.innerHTML = `<div style="color: #ffb347; text-align: center; padding: 50px;">${t.noEntityRelationships}</div>`;
        return;
    }

    try {
        window.triplesChart.setOption(option);
        console.log('Chart option set successfully');

        // Ensure chart is properly sized
        setTimeout(() => {
            if (window.triplesChart && typeof window.triplesChart.resize === 'function') {
                try {
                    window.triplesChart.resize();
                    console.log('Chart resized');
                } catch (e) {
                    console.warn('Error resizing chart:', e);
                }
            }
        }, 100);

        console.log('Triples chart rendered successfully!');
    } catch (e) {
        console.error('Error setting chart option:', e);
        chartContainer.innerHTML = `<div style="color: red; text-align: center; padding: 50px;">${t.chartRenderError}${e.message}</div>`;
    }
}

function extractSchemaType(entityStr) {
    const match = entityStr.match(/\[schema_type:\s*([^\]]+)\]/);
    return match ? match[1].trim() : null;
}

function cleanEntityName(entityStr) {
    return entityStr.replace(/\s*\[schema_type:[^\]]+\]/g, '').trim();
}

function getCategoryColor(category) {
    const key = String(category || '').trim().toLowerCase();
    const colors = {
        'person': '#74b9ff',
        'organization': '#8ed1ff',
        'location': '#6ee7b7',
        'event': '#fbbf24',
        'object': '#c4b5fd',
        'concept': '#60a5fa',
        'attribute': '#f472b6',
        'entity': '#93c5fd',
        'community': '#fb7185',
        'keyword': '#f59e0b',
        '\u4e3b\u9898\u793e\u533a': '#fb7185',
        '\u5173\u952e\u8bcd': '#f59e0b',
        '\u5c5e\u6027': '#f472b6',
        '\u8bba\u6587': '#34d399',
        '\u4f5c\u8005': '#74b9ff',
        '\u673a\u6784': '#22d3ee',
        '\u671f\u520a': '#f472b6',
        '\u7814\u7a76\u65b9\u6cd5': '#fb7185',
        '\u7814\u7a76\u4e3b\u9898': '#f59e0b',
        '\u6559\u80b2\u9886\u57df': '#a78bfa',
        '\u6559\u5b66\u573a\u666f': '#60a5fa',
        '\u6280\u672f': '#fbbf24'
    };
    return colors[key] || colors[String(category || '').trim()] || '#93c5fd';
}

function renderQueryChart(data) {
    const chartContainer = document.getElementById('queryChart');

    if (queryChart) {
        queryChart.dispose();
    }

    queryChart = echarts.init(chartContainer);

    const option = {
        backgroundColor: 'transparent',
        title: {
            text: 'Query Decomposition',
            left: 'center',
            textStyle: { color: 'rgba(255, 255, 255, 0.95)' }
        },
        tooltip: {
            trigger: 'item',
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            textStyle: { color: '#ffffff' }
        },
        series: [{
            type: 'graph',
            layout: 'force',
            data: data?.subqueries?.nodes || [
                {id: 'q1', name: 'Original Question', category: 'question', symbolSize: 40},
                {id: 'sq1', name: 'Sub-question 1', category: 'sub_question', symbolSize: 30},
                {id: 'sq2', name: 'Sub-question 2', category: 'sub_question', symbolSize: 30}
            ],
            links: data?.subqueries?.links || [
                {source: 'q1', target: 'sq1'},
                {source: 'q1', target: 'sq2'}
            ],
            categories: data?.subqueries?.categories || [
                {name: 'question', itemStyle: {color: '#6b73ff'}},
                {name: 'sub_question', itemStyle: {color: '#9bb5ff'}}
            ],
            roam: true,
            label: {
                show: true,
                color: 'rgba(255, 255, 255, 0.9)'
            },
            force: {
                repulsion: 800,
                gravity: 0.1
            },
            lineStyle: {
                opacity: 0.6,
                color: 'rgba(255, 255, 255, 0.4)'
            }
        }]
    };

    queryChart.setOption(option);
}

function showMessage(text, type = 'info', durationMs = 5000) {
    // Prefer QA-local container when present, fallback to global messages at bottom
    const local = document.getElementById('qaMessages');
    const messages = local || document.getElementById('messages');
    const message = document.createElement('div');
    message.className = `message ${type}`;
    message.textContent = text;

    messages.appendChild(message);

    setTimeout(() => {
        message.remove();
    }, durationMs);
}

function getMoreLabel(kind, count) {
    const t = i18n[currentLang] || i18n.en;
    // Reuse the same wording style for triples/chunks to keep i18n compact.
    return t.moreTriples(count);
}

window.toggleHiddenList = function(id, btn, kind = 'triples') {
    const t = i18n[currentLang] || i18n.en;
    const hiddenList = document.getElementById(id);
    if (!hiddenList) return;

    if (hiddenList.style.display === 'none') {
        hiddenList.style.display = 'block';
        btn.textContent = t.showLess;
    } else {
        hiddenList.style.display = 'none';
        const count = hiddenList.getElementsByTagName('li').length;
        btn.textContent = getMoreLabel(kind, count);
    }
}

// Backward compatibility for previous inline handlers.
window.toggleHiddenTriples = function(id, btn) {
    window.toggleHiddenList(id, btn, 'triples');
}
function renderReferences(papers, citedIndices) {
    if (!papers || !Array.isArray(papers) || papers.length === 0) return '';
    const sectionTitle = currentLang === 'zh' ? '参考文献' : 'References';
    const citedLabel   = currentLang === 'zh' ? '已引用' : 'Cited';

    /**
     * 按 GB/T 7714 格式拼接引文字符串
     * 格式：作者. 标题[J]. 期刊名, 年份.
     */
    function formatCitation(paper) {
        const parts = [];

        // 作者
        const authors = (paper.authors || '').trim();
        if (authors) parts.push(escapeHtml(authors));

        // 标题 + 文献类型标志
        const title = (paper.title || '').trim();
        if (title) {
            parts.push(escapeHtml(title) + '[J]');
        }

        // 期刊名
        const source = (paper.source || '').trim();
        // 年份
        const year = (paper.year || '').trim();

        let sourceYearStr = '';
        if (source && year) {
            sourceYearStr = `${escapeHtml(source)}, ${escapeHtml(year)}`;
        } else if (source) {
            sourceYearStr = escapeHtml(source);
        } else if (year) {
            sourceYearStr = escapeHtml(year);
        }
        if (sourceYearStr) parts.push(sourceYearStr);

        return parts.join('. ') + (parts.length ? '.' : '');
    }

    const items = papers.map(paper => {
        const isCited = citedIndices instanceof Set && citedIndices.has(paper.index);
        const citationStr = formatCitation(paper);

        return `
            <div class="reference-item${isCited ? ' ref-cited' : ''}" id="ref-${paper.index}">
                <div class="ref-header">
                    <span class="ref-index">[${paper.index}]</span>
                    <span class="ref-citation">${citationStr || escapeHtml(paper.title || 'Unknown')}</span>
                    ${isCited ? `<span class="ref-cited-badge">${citedLabel}</span>` : ''}
                </div>
            </div>
        `;
    }).join('');

    return `
        <div class="reference-section">
            <h4>${sectionTitle} <span class="ref-count-badge">${papers.length}</span></h4>
            <div class="reference-list">
                ${items}
            </div>
        </div>
    `;
}
