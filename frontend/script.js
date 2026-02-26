// Global variables
let selectedFiles = [];
let datasets = [];
let currentDataset = null;
let graphChart = null;
// let queryChart = null;
let questionCount = 0;

// i18n configuration: default to English; switch to Chinese when lang starts with 'zh'
let currentLang = (document.documentElement.lang || 'en').toLowerCase().startsWith('zh') ? 'zh' : 'en';
const i18n = {
    en: {
        processing: 'Processing your question...',
        pleaseSelectAndEnter: 'Please select a dataset and enter a question',
        subtitle: '✨ Vertically Unified Agents for Graph Retrieval-Augmented Complex Reasoning ✨',
        tabs: { dashboard: '📊 Dashboard', upload: '📤 Data Upload', graph: '🕸️ Graph Visualization', qa: '🤖 Q&A Interface' },
        systemOverview: 'System Overview',
        quickActions: 'Quick Actions',
        qaUploadDocsBtn: '📤 Upload Documents',
        qaViewGraphBtn: '🕸️ View Graph',
        qaQueryBtn: '🤖 Query',
        qaRefreshBtn: '🔄 Refresh',
        uploadDocuments: 'Upload Documents',
        uploadDragTitle: 'Drag & Drop Files Here',
        uploadDragDesc: 'or click to browse files',
        uploadSupports: 'Currently supports: .txt, .md, .json, .pdf, .docx, .doc',
        uploadSampleHeader: '📄 Sample Format (.json):',
        uploadBtn: 'Upload Files',
        clearUploadBtn: 'Clear',
        availableDatasets: 'Available Datasets',
        knowledgeTreeTitle: 'Knowledge Tree Visualization',
        labelDataset: 'Dataset:',
        selectDatasetPlaceholder: 'Select a dataset...',
        queryPanelTitle: 'Query Panel',
        labelQuery: 'Query:',
        askBtn: 'Ask Question',
        answerTitle: 'Answer:',
        decomposeStart: 'Starting to decompose your question into sub-questions...',
        decomposeSummary: (t) => `Decomposition complete. Total sub-questions: ${t}`,
        subQuestion: (i,t,q) => `Sub-question ${i}/${t}: ${q}`,
        ircotStart: 'Starting IRCoT reasoning...',
        ircotStep: (s,max,thought) => `IRCoT Step ${s}/${max}: ${thought}`,
        retrievalProgress: (found,cand) => `Retrieval progress: ${found} relevant nodes from ${cand} candidates.`,
        answerStart: 'Synthesizing the final answer...',
        qaCompleted: 'QA process completed!',
        completedBadge: '✔️ Completed',
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
        processing: '正在处理你的问题…',
        pleaseSelectAndEnter: '请选择数据集并输入问题',
        subtitle: '✨ 面向图检索增强复杂推理的统一式智能体 ✨',
        tabs: { dashboard: '📊 仪表盘', upload: '📤 数据上传', graph: '🕸️ 图谱可视化', qa: '🤖 问答接口' },
        systemOverview: '系统概览',
        quickActions: '快捷操作',
        qaUploadDocsBtn: '📤 上传文档',
        qaViewGraphBtn: '🕸️ 查看图谱',
        qaQueryBtn: '🤖 发起查询',
        qaRefreshBtn: '🔄 刷新',
        uploadDocuments: '上传文档',
        uploadDragTitle: '拖拽文件到这里',
        uploadDragDesc: '或点击选择文件',
        uploadSupports: '当前支持：.txt、.md、.json、.pdf、.docx、.doc',
        uploadSampleHeader: '📄 示例格式（.json）：',
        uploadBtn: '开始上传',
        clearUploadBtn: '清空',
        availableDatasets: '可用数据集',
        knowledgeTreeTitle: '知识树可视化',
        labelDataset: '数据集：',
        selectDatasetPlaceholder: '请选择数据集…',
        queryPanelTitle: '查询面板',
        labelQuery: '查询：',
        askBtn: '发起提问',
        answerTitle: '答案：',
        decomposeStart: '开始将你的问题分解为子问题…',
        decomposeSummary: (t) => `分解完成。子问题总数：${t}`,
        subQuestion: (i,t,q) => `子问题 ${i}/${t}：${q}`,
        ircotStart: '开始 IRCoT 推理…',
        ircotStep: (s,max,thought) => `IRCoT 步骤 ${s}/${max}：${thought}`,
        retrievalProgress: (found,cand) => `检索进度：在 ${cand} 个候选中找到 ${found} 个相关节点。`,
        answerStart: '正在综合生成最终答案…',
        qaCompleted: '问答流程完成！',
        completedBadge: '✔️ 已完成',
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
    const labelGraphDataset = document.getElementById('labelGraphDataset'); if (labelGraphDataset) labelGraphDataset.textContent = t.labelDataset;
    const graphSelect = document.getElementById('graphDataset'); if (graphSelect) graphSelect.options[0].text = t.selectDatasetPlaceholder;
    // QA
    const queryPanelTitle = document.getElementById('queryPanelTitle'); if (queryPanelTitle) queryPanelTitle.textContent = t.queryPanelTitle;
    const labelQADataset = document.getElementById('labelQADataset'); if (labelQADataset) labelQADataset.textContent = t.labelDataset;
    const qaSelect = document.getElementById('qaDataset'); if (qaSelect) qaSelect.options[0].text = t.selectDatasetPlaceholder;
    const labelQuery = document.getElementById('labelQuery'); if (labelQuery) labelQuery.textContent = t.labelQuery;
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

function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    event.target.classList.add('active');

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
            progressText.textContent = `Uploading... ${Math.round(progress)}%`;
        }, 500);

        const response = await axios.post(`${API_BASE}/api/upload`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });

        clearInterval(progressInterval);
        progressFill.style.width = '100%';
        progressText.textContent = 'Upload completed!';

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
        showMessage('Upload failed: ' + (error.response?.data?.detail || error.message), 'error');
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
        showMessage('Failed to load datasets', 'error');
    }
}

function displayDatasets() {
    const container = document.getElementById('datasetsList');
    if (!container) return;
    if (datasets.length === 0) {
        container.innerHTML = '<p style="color: rgba(255,255,255,0.6);">No datasets available. Upload some files to get started.</p>';
        return;
    }

    container.innerHTML = datasets.map(dataset => `
        <div class="file-item">
            <div>
                <strong>${formatDatasetDisplayName(dataset)}</strong>
                <span style="color: rgba(255,255,255,0.6);"> (${dataset.type})</span>
                ${dataset.type !== 'demo' ? `<span style="margin-left:8px; font-size:12px; color:${dataset.has_custom_schema ? '#228B22' : 'rgba(255,255,255,0.6)'}; font-weight:600;">Schema: ${dataset.has_custom_schema ? 'custom' : 'default'}</span>` : ''}
            </div>
            <div class="dataset-actions">
                <span class="status-badge status-${dataset.status === 'ready' ? 'ready' : dataset.status === 'constructing' ? 'processing' : 'error'}">
                    ${dataset.status}
                </span>
                ${dataset.status === 'needs_construction' ? 
                    `<button class="btn btn-primary" onclick="constructGraph('${dataset.name}')">Construct</button>` : 
                    ''
                }
                ${dataset.status === 'ready' ? 
                    `<button class="btn btn-secondary" onclick="reconstructGraph('${dataset.name}')" title="Reconstruct Graph">🔄 Reconstruct</button>` : 
                    ''
                }
                ${dataset.type !== 'demo' ? 
                    `<button class="btn" onclick="triggerSchemaUpload('${dataset.name}')" title="Upload custom schema">📚 Upload Schema</button>` : 
                    ''
                }
                ${dataset.type !== 'demo' ? 
                    `<button class="btn btn-danger" onclick="deleteDataset('${dataset.name}')" title="Delete Dataset">🗑️ Delete</button>` : 
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
                showMessage(`Uploading schema for "${datasetName}"...`, 'info');
                const fd = new FormData();
                fd.append('schema_file', file);
                await axios.post(`${API_BASE}/api/datasets/${datasetName}/schema`, fd, {
                    headers: { 'Content-Type': 'multipart/form-data' }
                });
                showMessage('Schema uploaded successfully!', 'success');
                await loadDatasets();
            } catch (err) {
                console.error('Schema upload failed:', err);
                showMessage('Schema upload failed: ' + (err.response?.data?.detail || err.message), 'error');
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
        showMessage('Starting graph construction...', 'info');

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
                    showMessage(`[Construct ${progressMessages.length}] ${data.message}`, 'info');
                } else if (data.type === 'complete') {
                    showMessage('Graph construction completed!', 'success');
                    // refresh datasets immediately to reflect ready status
                    refreshData();
                    ws.close();
                } else if (data.type === 'error') {
                    showMessage(`Construction error: ${data.message}`, 'error');
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
                    showMessage('Graph construction completed!', 'success');
                }
            }
        }, 30000); // 30s timeout

        // 不再立即 refreshData，等 WebSocket complete/error 后再刷新
    } catch (error) {
        console.error('Construction failed:', error);
        showMessage('Graph construction failed: ' + (error.response?.data?.detail || error.message), 'error');
    }
}

async function reconstructGraph(datasetName) {
    if (!confirm(`Are you sure you want to reconstruct the graph for dataset "${datasetName}"? This will delete the existing graph and cache files.`)) {
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
        showMessage('Reconstructing graph...', 'info');

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
                    showMessage(`[Reconstruct ${progressMessages.length}] ${data.message}`, 'info');
                } else if (data.type === 'complete') {
                    showMessage('Graph reconstruction completed!', 'success');
                    // refresh datasets immediately to reflect ready status
                    refreshData();
                    ws.close();
                } else if (data.type === 'error') {
                    showMessage(`Reconstruction error: ${data.message}`, 'error');
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
                    showMessage('Graph reconstruction completed!', 'success');
                }
            }
        }, 30000); // 30s timeout

        // 不再立即 refreshData，等 WebSocket complete/error 后再刷新
    } catch (error) {
        console.error('Reconstruction failed:', error);
        showMessage('Graph reconstruction failed: ' + (error.response?.data?.detail || error.message), 'error');
    }
}

async function deleteDataset(datasetName) {
    if (!confirm(`Are you sure you want to delete dataset "${datasetName}"? This action will remove all related files and cannot be undone.`)) {
        return;
    }

    try {
        showMessage('Deleting dataset...', 'info');
        const response = await axios.delete(`${API_BASE}/api/datasets/${datasetName}`);
        showMessage(`Dataset "${datasetName}" deleted successfully!`, 'success');
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
        showMessage('Dataset deletion failed: ' + (error.response?.data?.detail || error.message), 'error');
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
        showMessage('Failed to load graph data', 'error');
    }
}

function renderGraph(data) {
    const chartContainer = document.getElementById('graphChart');

    if (graphChart) {
        graphChart.dispose();
    }

    graphChart = echarts.init(chartContainer);

    const option = {
        backgroundColor: 'transparent',
        // title: {
        //     text: 'Knowledge Graph',
        //     left: 'center',
        //     textStyle: { color: 'rgba(255, 255, 255, 0.95)' }
        // },
        tooltip: {
            trigger: 'item',
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            textStyle: { color: '#ffffff' },
            formatter: function (params) {
                if (params.dataType === 'node') {
                    const d = params.data || {};
                    let name = (d.name || '').toString().replace(/\s+/g,' ').trim();
                    if (name.length > 20) name = name.slice(0,20) + '...';
                    const category = d.category || d.type || '';
                    return name ? name : category || 'node';
                } else if (params.dataType === 'edge') {
                    return params.data && params.data.name ? params.data.name : '';
                }
                return '';
            }
        },
        legend: {
            type: 'scroll',
            bottom: 10,
            textStyle: { color: 'rgba(255, 255, 255, 0.85)' },
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
                show: true,
                color: 'rgba(255, 255, 255, 0.9)',
                formatter: function(p){
                    const d = p.data || {};
                    let name = (d.name || '').toString().replace(/\s+/g,' ').trim();
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
                opacity: 0.6,
                color: 'rgba(255, 255, 255, 0.4)'
            }
        }]
    };

    graphChart.setOption(option);
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
                    if (data.stage === 'sub_question') {
                        const q = (data.question || '').toString();
                        if (currentLang === 'zh') {
                            const msg = `【子问题 ${data.index}/${data.total}】${q}｜三元组：${data.triples_count || 0}｜片段：${data.chunks_count || 0}`;
                            showMessage(msg, 'info');
                            if (Array.isArray(data.triples_preview) && data.triples_preview.length) {
                                showMessage('预览：' + data.triples_preview.slice(0,3).join(' | '), 'info');
                            }
                        } else {
                            const msg = `[Sub ${data.index}/${data.total}] ${q} | Triples: ${data.triples_count || 0} | Chunks: ${data.chunks_count || 0}`;
                            showMessage(msg, 'info');
                            if (Array.isArray(data.triples_preview) && data.triples_preview.length) {
                                showMessage('Preview: ' + data.triples_preview.slice(0,3).join(' | '), 'info');
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
        showMessage('Failed to process question: ' + (error.response?.data?.detail || error.message), 'error');
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

function displayAnswer(result) {
    console.log('displayAnswer called with result:', result);

    document.getElementById('answerContent').textContent = result.answer;

    // Display detailed retrieval information
    displayRetrievalDetails(result);

    // Render query analysis chart
    // renderQueryChart(result.visualization_data);

    document.getElementById('answerSection').classList.remove('hidden');
}

function displayRetrievalDetails(result) {
    console.log('displayRetrievalDetails called with:', result);

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
    <h4>🔍 Retrieval Stats</h4>
            <div class="stats-grid">
                <div class="stat-item">
        <span class="stat-label">Sub-questions:</span>
                    <span class="stat-value">${result.sub_questions?.length || 0}</span>
                </div>
                <div class="stat-item">
        <span class="stat-label">Retrieved Triples (Sum):</span>
                    <span class="stat-value">${subQuestionTriplesTotal}</span>
                </div>
                <div class="stat-item">
        <span class="stat-label">Relevant Chunks:</span>
                    <span class="stat-value">${result.retrieved_chunks?.length || 0}</span>
                </div>
            </div>
        </div>
        
        <div class="subquestions-section">
    <h4>📝 Question Decomposition</h4>
            <div class="subquestions-list">
    `;

    if (result.sub_questions && result.sub_questions.length > 0) {
        result.sub_questions.forEach((sq, index) => {
            const step = result.reasoning_steps?.[index] || {};
            const stepTriples = dedupTriples(step.triples || []);

            // 1. 生成新的 HTML 逻辑
            const visibleTriples = stepTriples.slice(0, 3);
            const hiddenTriples = stepTriples.slice(3);

            let triplesHtml = '';
            if (stepTriples.length > 0) {
                triplesHtml = `
                    <div class="triples-preview">
                        <strong>Retrieved Triples:</strong>
                        <ul>
                            ${visibleTriples.map(triple => `<li>${triple}</li>`).join('')}
                        </ul>
                        
                        ${hiddenTriples.length > 0 ? `
                            <ul id="hidden-triples-${index}" style="display: none; margin-top: 0;">
                                ${hiddenTriples.map(triple => `<li>${triple}</li>`).join('')}
                            </ul>
                            <span class="more-indicator" 
                                  style="cursor: pointer; text-decoration: underline; color: #3b82f6;" 
                                  onclick="toggleHiddenTriples('hidden-triples-${index}', this)">
                                ...and ${hiddenTriples.length} more
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
                        <span>Triples: ${stepTriples.length}</span>
                        <span>Chunks: ${step.chunks_count || 0}</span>
                        <span>Time: ${(step.processing_time || 0).toFixed(2)}s</span>
                    </div>
                    ${triplesHtml} 
                </div>
            `;
        });
    } else {
        detailsHtml += '<p class="no-data">No sub-question data</p>';
    }

    detailsHtml += `
            </div>
        </div>
        
        <div class="triples-section">
            <h4>🔗 Subgraph Visualization</h4>
            <div class="triples-chart-container">
                <div id="triplesChart" class="chart-container" style="height: 400px;"></div>
            </div>
            <div class="triples-summary">
                <p>Retrieved <strong>${result.retrieved_triples?.length || 0}</strong> related triples</p>
            </div>
        </div>
    `;

    detailsHtml += '</div></div>';
    detailsContainer.innerHTML = detailsHtml;

    // Render triples chart after setting HTML and ensuring container is visible
    setTimeout(() => {
        console.log('About to render triples chart with:', result.retrieved_triples?.length || 0, 'triples');
        // Make sure the chart container is visible before initializing ECharts
        const chartContainer = document.getElementById('triplesChart');
        if (chartContainer) {
            chartContainer.style.display = 'block';
            console.log('Chart container made visible');
        }
        renderTriplesChart(result.retrieved_triples || []);
    }, 200);
}

function renderTriplesChart(triples) {
    console.log('renderTriplesChart called with:', triples.length, 'triples');
    console.log('First triple example:', triples[0]);

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
        chartContainer.innerHTML = '<div style="color: red; text-align: center; padding: 50px;">ECharts library not loaded</div>';
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
        chartContainer.innerHTML = '<div style="color: red; text-align: center; padding: 50px;">Error initializing ECharts: ' + e.message + '</div>';
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
                    color: '#6b73ff',
                    width: 2
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
        backgroundColor: 'transparent',
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
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            textStyle: { color: '#ffffff' },
            formatter: function(params) {
                if (params.dataType === 'node') {
                    const rawFull = (params.data.rawName || params.data.id || '').toString().replace(/\s+/g,' ').trim();
                    return `<strong>${rawFull}</strong><br/>Type: ${params.data.category}`;
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
            textStyle: { color: 'rgba(255, 255, 255, 0.85)' },
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
                color: 'rgba(255, 255, 255, 0.9)',
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
                opacity: 0.8,
                curveness: 0.1
            },
            edgeLabel: {
                show: true,
                color: 'rgba(255,255,255,0.75)',
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
        chartContainer.innerHTML = '<div style="color: #ffb347; text-align: center; padding: 50px;">No entity relationships found</div>';
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
        chartContainer.innerHTML = '<div style="color: red; text-align: center; padding: 50px;">Chart rendering error: ' + e.message + '</div>';
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
    const colors = {
        'person': '#6b73ff',
        'organization': '#9bb5ff',
        'location': '#a8c8ec',
        'event': '#b8d4f0',
        'object': '#e6f3ff',
        'concept': '#4a90e2',
        'attribute': '#6b73ff',
        'entity': '#9bb5ff'
    };
    return colors[category] || '#9bb5ff';
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

// 添加在 script.js 末尾
window.toggleHiddenTriples = function(id, btn) {
    const hiddenList = document.getElementById(id);
    if (hiddenList) {
        // 切换显示状态
        if (hiddenList.style.display === 'none') {
            hiddenList.style.display = 'block';
            btn.textContent = 'Show less'; // 点击后文案变成收起
        } else {
            hiddenList.style.display = 'none';
            // 恢复显示 "...and X more"
            const count = hiddenList.getElementsByTagName('li').length;
            btn.textContent = `...and ${count} more`;
        }
    }
}