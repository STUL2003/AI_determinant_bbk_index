import aspose.pdf as ap
import re
import os

class pdf2tex:
    """Класс для конвертации из PDF в LaTex"""
    def __init__(self, filePdf):
        self._inputFile = filePdf
        self._textLatex = None

    @property
    def textLatex(self):
        if self._textLatex is None:
            self._convert_pdf2latex()
            self._preprocess_for_mathbert()
        return self._textLatex

    def _convert_pdf2latex(self):
        output_file = self._inputFile.split('.')[0] + ".tex"
        document = ap.Document(self._inputFile)
        save_options = ap.LaTeXSaveOptions()
        document.save(output_file, save_options)

        with open(output_file, "r", encoding="utf-8") as f:
            self._textLatex = f.read()

        os.remove(output_file)

    def _preprocess_for_mathbert(self):

        #замена разделителей формул
        processed = re.sub(r'\$(.*?)\$', r'\(\1\)', self._textLatex, flags=re.DOTALL)

        #удаление LaTeX-комментариев
        processed = re.sub(r'\\%.*', '', processed)
        processed =re.sub(  #очистка лишних пробелов внутри формул
            r'\\(\()\s*(.*?)\s*\\()\)',
            r'\(\2\)',
            processed,
            flags=re.DOTALL
        )
        self._textLatex = processed