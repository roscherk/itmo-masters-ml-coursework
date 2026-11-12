export default function () {
  return {
    strict: false,
    macros: {
      // множества
      '\\R': '\\mathbb{R}',
      '\\N': '\\mathbb{N}',
      '\\Z': '\\mathbb{Z}',
      '\\Q': '\\mathbb{Q}',
      '\\C': '\\mathbb{C}',

      // русская типографика неравенств
      '\\le': '\\leqslant',
      '\\ge': '\\geqslant',

      // в KaTeX \ensuremath не нужен (он всегда в math mode), но пусть будет
      '\\ensuremath': '#1',

      // операторы
      '\\argmin': '\\operatorname*{arg\\,min}',
      '\\argmax': '\\operatorname*{arg\\,max}',

      // скобки
      '\\abs': '\\left|#1\\right|',
      '\\norm': '\\left\\lVert#1\\right\\rVert',
      '\\inner': '\\left\\langle #1, #2 \\right\\rangle',

      // буквы
      '\\eps': '\\varepsilon',
    },
  }
}
