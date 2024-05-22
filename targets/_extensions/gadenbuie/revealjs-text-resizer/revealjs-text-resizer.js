window.RevealTextResizer = function () {
  return {
    id: "RevealTextResizer",
    init: function () {
      document.addEventListener('keydown', function(event) {
        if (!event.altKey) return;
        const go_up = event.code === 'Equal';
        const go_down = event.code === 'Minus';
        const go_reset = event.code === 'Digit0';

        if (!(go_up || go_down || go_reset)) return;
        event.preventDefault();

        if (go_reset) {
          document.documentElement.style.removeProperty('--r-main-font-size');
          return;
        }

        const currentSize = parseFloat(
          getComputedStyle(document.documentElement)
            .getPropertyValue('--r-main-font-size')
        );

        const newSize = go_up ? currentSize + 2 : currentSize - 2;
        document.documentElement.style.setProperty('--r-main-font-size', newSize + 'px');
      });
    },
  };
};

