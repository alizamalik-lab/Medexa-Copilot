import { marked } from 'marked';

const CODE_PATTERN = /^(?:[0-9]{5}|G[0-9]{4}|[A-TV-Z][0-9][0-9AB]\.?[0-9A-Z]{0,4})$/i;

const renderer = new marked.Renderer();
const baseCodespan = renderer.codespan.bind(renderer);

renderer.codespan = (token) => {
  const raw = (typeof token === 'string' ? token : token?.text ?? '').trim();

  if (CODE_PATTERN.test(raw)) {
    return `<span class="code-chip" data-code="${raw}" title="Click to copy">${raw}</span>`;
  }

  return baseCodespan(token);
};

marked.setOptions({ renderer, breaks: true });
