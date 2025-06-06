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
var modal1 = document.getElementById("myModal");

// Get the button that opens the modal
var btn1 = document.getElementById("check--keyterm");

// Get the <span> element that closes the modal
var span1 = document.getElementsByClassName("close")[0]; // первый элемент закрытия

var modal2 = document.getElementById("myModal2");
var btn2 = document.getElementById("check--indexinfo");
var span2 = document.getElementsByClassName("close")[1]; // второй элемент закрытия

var modal3 = document.getElementById("myModal3");
var btn3 = document.getElementById("send--data");
var span3 = document.getElementsByClassName("close")[2]; // второй элемент закрытия

// When the user clicks the button, open the modal
btn1.onclick = async function() {
  modal1.style.display = "block";
  const response = await fetch('/static/data/keywords_impact.txt');
  const data = await response.text();
  document.getElementById("model--body").innerHTML = data.replace(/\n/g, '<br>');
};
// When the user clicks on <span> (x), close the modal
span1.onclick = function() {
  modal1.style.display = "none";
}

btn2.onclick = async function() {
  modal2.style.display = "block";

  const response = await fetch('/static/data/results.json');
  const data = await response.json();


  let content = `<div class="hierarchy-tree">`;

    // Перебираем все уровни иерархии
  data.tree.forEach(level => {
    const isSecondary = level.is_secondary || false;
    const secondaryClass = isSecondary ? "secondary-branch" : "";

    content += `
      <div class="tree-level ${secondaryClass}">
        <div class="level-header">
          <span class="level-title">Уровень ${level.level}${isSecondary ? " (дополнительная ветка)" : ""}</span>
        </div>
        <div class="level-content">
          <ul class="level-results">`;

      // Перебираем все результаты в этом уровне
    level.results.forEach(item => {
      const score=item.score ? item.score.toFixed(4) : "N/A";
      content += `
        <li>
          <span class="code">${item.code}</span>
          <span class="name">${item.name}</span>
          <span class="score">${score}</span>
        </li>`;
    });

    content += `
          </ul>
        </div>
      </div>`;
  });

  content += `</div>`;
  document.getElementById("model--body2").innerHTML = content;

  const headers = document.querySelectorAll('.level-header');
  headers.forEach(header => {
    header.addEventListener('click', function() {
      const content = this.nextElementSibling;
      const icon = this.querySelector('.toggle-icon');
      if (content.style.display === "block") {
        content.style.display = "none";
        icon.textContent = "►";
      } else {
        content.style.display = "block";
        icon.textContent = "▼";
      }
    });
  });
};

span2.onclick = function() {
  modal2.style.display = "none";
}

btn3.onclick = function(){
    modal3.style.display = "block";
}

span3.onclick = function() {
  modal3.style.display = "none";
}
// When the user clicks anywhere outside of the modal, close it
window.onclick = function(event) {
  if (event.target == modal1) {
    modal1.style.display = "none";
  }
  if (event.target == modal2) {
    modal2.style.display = "none";
  }
  if (event.target == modal3) {
    modal3.style.display = "none";
  }
}

document.getElementById('send-index').addEventListener('submit', function(e) {
    e.preventDefault();

    const formData = new FormData(this);

    fetch('/trainindexes', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Показываем сообщение без перезагрузки
        showFlashMessage(data.message || "Данные отправлены на дообучение");
    })
    .catch(error => {
        showFlashMessage("Ошибка: " + error.message);
    });
});

function showFlashMessage(message) {
    const flash = document.createElement('div');
    flash.className = 'flash-message';
    flash.textContent = message;
    document.body.appendChild(flash);

    setTimeout(() => {
        flash.remove();
    }, 5000);
}