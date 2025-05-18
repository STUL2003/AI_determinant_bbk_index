let fileName = null;
let isProcessing = false;
let isAnalyze = false;

document.getElementById('file').addEventListener('change', async function(e) {
    const file = e.target.files[0];
    fileName = file.name;

    document.getElementById('file-name').textContent = fileName;
    document.getElementById('close--one').textContent = '✖';

    isProcessing = true;
    infoLoad();
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch('/extract-text', {
        method: 'POST',
        body: formData
    });
    const text = await response.text();
    document.getElementById('book-text').textContent = text;
    isProcessing = false;

});

async function infoLoad() {// Колхоз, но пофиг
    document.getElementById('book-list').textContent = '';
    while (isProcessing) {
        await new Promise(r => setTimeout(r, 500));
        if (!isProcessing) break;
        document.getElementById('book-text').textContent = 'Текст загружается';
        await new Promise(r => setTimeout(r, 500));
        if (!isProcessing) break;
        document.getElementById('book-text').textContent = 'Текст загружается .';
        await new Promise(r => setTimeout(r, 500));
        if (!isProcessing) break;
        document.getElementById('book-text').textContent = 'Текст загружается . .';
        await new Promise(r => setTimeout(r, 500));
        if (!isProcessing) break;
        document.getElementById('book-text').textContent = 'Текст загружается . . .';
    }
}

async function analyzeLoad() {// Не получилось
    while (isAnalyze) {
        const response = await fetch('/check_analyze');
        const data = await response.json();

        if (!data.is_analyzing) {
            isAnalyze = false;
            break;
        }
        await new Promise(r => setTimeout(r, 500));
        if (!isAnalyze) break;
        document.getElementById('book-list').textContent = 'Текст анализируется';
        await new Promise(r => setTimeout(r, 500));
        if (!isAnalyze) break;
        document.getElementById('book-list').textContent = 'Текст анализируется .';
        await new Promise(r => setTimeout(r, 500));
        if (!isAnalyze) break;
        document.getElementById('book-list').textContent = 'Текст анализируется . .';
        await new Promise(r => setTimeout(r, 500));
        if (!isAnalyze) break;
        document.getElementById('book-list').textContent = 'Текст анализируется . . .';
    }
}

document.getElementById('close--one').onclick = function(){
    isProcessing = false;
    document.getElementById('file-name').textContent = '';
    document.getElementById('close--one').textContent = '';
    document.getElementById('file').value = '';
    document.getElementById('book-text').textContent = '';
};

document.getElementById('upload-form').addEventListener('submit', async function(e) {
    document.querySelectorAll('.block, #file').forEach(el => {
        el.style.pointerEvents = 'none';
        el.style.opacity = '0.5';
    });

    try {
    } catch {
        document.querySelectorAll('.block, #file').forEach(el => {
            el.style.pointerEvents = 'auto';
            el.style.opacity = '1';
        });
    }
});



// Get the modal
var modal = document.getElementById("myModal");

// Get the button that opens the modal
var btn = document.getElementById("check--keyterm");

// Get the <span> element that closes the modal
var span = document.getElementsByClassName("close")[0];

// When the user clicks the button, open the modal 
btn.onclick = async function() {
  modal.style.display = "block";
  const response = await fetch('/static/data/keywords_impact.txt');
  const data = await response.text();
  document.getElementById("model--body").textContent = data;
};
// When the user clicks on <span> (x), close the modal
span.onclick = function() {
  modal.style.display = "none";
}

// When the user clicks anywhere outside of the modal, close it
window.onclick = function(event) {
  if (event.target == modal) {
    modal.style.display = "none";
  }
}