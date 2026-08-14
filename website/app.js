/**
 * PEFT Benchmarking — Interactive UI Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // Table Filtering
  const modelFilters = document.querySelectorAll('[data-model-filter]');
  const datasetFilters = document.querySelectorAll('[data-dataset-filter]');
  const tableRows = document.querySelectorAll('.benchmark-table tbody tr');

  let activeModel = 'all';
  let activeDataset = 'all';

  function filterTable() {
    tableRows.forEach(row => {
      const rowModel = row.getAttribute('data-model');
      const rowDataset = row.getAttribute('data-dataset');

      const matchModel = (activeModel === 'all' || rowModel === activeModel);
      const matchDataset = (activeDataset === 'all' || rowDataset === activeDataset);

      if (matchModel && matchDataset) {
        row.style.display = '';
      } else {
        row.style.display = 'none';
      }
    });
  }

  modelFilters.forEach(btn => {
    btn.addEventListener('click', () => {
      modelFilters.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeModel = btn.getAttribute('data-model-filter');
      filterTable();
    });
  });

  datasetFilters.forEach(btn => {
    btn.addEventListener('click', () => {
      datasetFilters.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeDataset = btn.getAttribute('data-dataset-filter');
      filterTable();
    });
  });

  // Dynamic Nav theme switching based on active section
  const sections = document.querySelectorAll('section[data-surface]');
  const topNav = document.querySelector('.top-nav');

  function updateNavTheme() {
    const scrollPos = window.scrollY + 60;
    sections.forEach(sec => {
      const top = sec.offsetTop;
      const height = sec.offsetHeight;
      if (scrollPos >= top && scrollPos < top + height) {
        const surface = sec.getAttribute('data-surface');
        if (surface === 'dark') {
          topNav.classList.add('dark-mode');
        } else {
          topNav.classList.remove('dark-mode');
        }
      }
    });
  }

  window.addEventListener('scroll', updateNavTheme);
  updateNavTheme();
});
