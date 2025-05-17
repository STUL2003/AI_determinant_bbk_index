let fileName = null;
let isProcessing = false;

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
document.getElementById('close--one').onclick = function(){
    isProcessing = false;
    document.getElementById('file-name').textContent = '';
    document.getElementById('close--one').textContent = '';
    document.getElementById('file').value = '';
    document.getElementById('book-text').textContent = '';
};