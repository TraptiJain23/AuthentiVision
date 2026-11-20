// AuthentiVision Client-Side Dashboard Controller

document.addEventListener("DOMContentLoaded", function () {
    // 1. Setup Drag and Drop Zones
    const dropzones = document.querySelectorAll(".dropzone-container");

    dropzones.forEach(dropzone => {
        const fileInput = dropzone.querySelector("input[type='file']");
        const previewContainer = dropzone.querySelector(".file-preview-info");

        if (!fileInput) return;

        dropzone.addEventListener("click", () => fileInput.click());

        dropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropzone.classList.add("dragover");
        });

        ["dragleave", "dragend"].forEach(type => {
            dropzone.addEventListener(type, () => dropzone.classList.remove("dragover"));
        });

        dropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropzone.classList.remove("dragover");
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                handleFileSelection(fileInput, previewContainer);
            }
        });

        fileInput.addEventListener("change", () => {
            handleFileSelection(fileInput, previewContainer);
        });
    });

    function handleFileSelection(input, previewContainer) {
        if (!previewContainer) return;
        if (input.files.length === 1) {
            const file = input.files[0];
            previewContainer.innerHTML = `
                <div class="alert alert-primary py-2 px-3 mt-3 d-inline-flex align-items-center gap-2">
                    <i class="fa-solid fa-file-image"></i>
                    <span class="fw-semibold">${file.name}</span>
                    <span class="badge bg-secondary">${(file.size / (1024 * 1024)).toFixed(2)} MB</span>
                </div>
            `;
        } else if (input.files.length > 1) {
            previewContainer.innerHTML = `
                <div class="alert alert-primary py-2 px-3 mt-3 d-inline-flex align-items-center gap-2">
                    <i class="fa-solid fa-folder-open"></i>
                    <span class="fw-semibold">${input.files.length} assets selected for batch audit</span>
                </div>
            `;
        }
    }

    // 2. Form submission loading indicators
    const scanForms = document.querySelectorAll(".scan-form");
    scanForms.forEach(form => {
        form.addEventListener("submit", function () {
            const submitBtn = form.querySelector("button[type='submit']");
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = `
                    <span class="spinner-border spinner-border-sm me-2" role="status"></span>
                    <span>Analyzing Image...</span>
                `;
            }
        });
    });
});

// Helper to initialize Campaign Batch Chart
function renderCampaignCharts(realCount, aiCount, avgConf) {
    const pieCtx = document.getElementById("campaignPieChart");
    if (pieCtx) {
        new Chart(pieCtx, {
            type: "doughnut",
            data: {
                labels: ["Authentic Assets", "AI-Generated / Synthetic"],
                datasets: [{
                    data: [realCount, aiCount],
                    backgroundColor: ["#10b981", "#ef4444"],
                    borderWidth: 2,
                    borderColor: "#ffffff"
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "bottom",
                        labels: { boxWidth: 14, font: { family: 'Plus Jakarta Sans', weight: '600' } }
                    }
                },
                cutout: "65%"
            }
        });
    }
}
