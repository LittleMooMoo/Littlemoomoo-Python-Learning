(() => {
  'use strict';
  const toggle = document.querySelector('.nav-toggle');
  const nav = document.querySelector('.nav-links');
  toggle?.addEventListener('click', () => {
    const open = toggle.getAttribute('aria-expanded') !== 'true';
    toggle.setAttribute('aria-expanded', String(open));
    toggle.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
    nav.classList.toggle('is-open', open);
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && toggle) {
      toggle.setAttribute('aria-expanded', 'false');
      toggle.setAttribute('aria-label', 'Open navigation');
      nav.classList.remove('is-open');
    }
  });
  document.querySelectorAll('[data-filter]').forEach(button => {
    button.addEventListener('click', () => {
      document.querySelectorAll('[data-filter]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
      const filter = button.dataset.filter;
      document.querySelectorAll('[data-category]').forEach(card => {
        card.hidden = filter !== 'all' && !card.dataset.category.split(' ').includes(filter);
      });
      const status = document.getElementById('filter-status');
      if (status) status.textContent = `${document.querySelectorAll('[data-category]:not([hidden])').length} projects shown`;
    });
  });
  const lightbox = document.querySelector('.lightbox');
  let opener;
  if (lightbox) {
    const picture = lightbox.querySelector('img');
    const caption = lightbox.querySelector('.lightbox-caption');
    document.querySelectorAll('[data-lightbox]').forEach(button => {
      button.addEventListener('click', () => {
        opener = button;
        const source = button.querySelector('img');
        picture.src = source.src;
        picture.alt = source.alt;
        caption.textContent = button.dataset.caption || source.alt;
        lightbox.showModal();
        document.body.classList.add('modal-open');
      });
    });
    lightbox.querySelector('.lightbox-close').addEventListener('click', () => lightbox.close());
    lightbox.addEventListener('click', event => {
      const box = lightbox.getBoundingClientRect();
      if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) lightbox.close();
    });
    lightbox.addEventListener('close', () => {
      document.body.classList.remove('modal-open');
      opener?.focus();
    });
  }
  document.querySelectorAll('[data-copy-email]').forEach(button => {
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(button.dataset.copyEmail);
        button.textContent = 'Copied';
        window.setTimeout(() => { button.textContent = 'Copy email'; }, 2200);
      } catch {
        button.textContent = 'Select the email to copy';
      }
    });
  });
  document.querySelector('[data-print]')?.addEventListener('click', () => window.print());
})();
