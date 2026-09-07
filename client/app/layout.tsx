import { Analytics } from '@vercel/analytics/next'
import { Geist, Geist_Mono } from 'next/font/google'
import type { Metadata, Viewport } from 'next'
import './globals.css'

const geist = Geist({ subsets: ['latin'], variable: '--font-geist' })
const geistMono = Geist_Mono({ subsets: ['latin'], variable: '--font-geist-mono' })

export const metadata: Metadata = { title: 'SatQuery AI — Ask your satellite imagery anything', description: 'Agentic vision-language analysis for remote-sensing imagery.', generator: 'SatQuery AI' }
export const viewport: Viewport = { colorScheme: 'dark light', themeColor: '#090a0e' }
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`bg-background ${geist.variable} ${geistMono.variable}`}
      data-scroll-behavior="smooth"
      suppressHydrationWarning
    >
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                // 1. Intercept extension runtime errors from triggering Next.js dev error overlay
                window.addEventListener('error', function(event) {
                  var file = event.filename || '';
                  var stack = (event.error && event.error.stack) || '';
                  var msg = event.message || '';
                  if (
                    file.indexOf('chrome-extension://') !== -1 ||
                    stack.indexOf('chrome-extension://') !== -1 ||
                    msg.indexOf('M_ID') !== -1
                  ) {
                    event.stopImmediatePropagation();
                    event.preventDefault();
                    return true;
                  }
                }, true);

                window.addEventListener('unhandledrejection', function(event) {
                  var reason = event.reason;
                  var stack = (reason && reason.stack) || '';
                  var msg = (reason && reason.message) || String(reason || '');
                  if (
                    stack.indexOf('chrome-extension://') !== -1 ||
                    msg.indexOf('chrome-extension://') !== -1 ||
                    msg.indexOf('M_ID') !== -1
                  ) {
                    event.stopImmediatePropagation();
                    event.preventDefault();
                  }
                }, true);

                // 2. Filter console.error for extension hydration artifacts
                var _origError = console.error;
                console.error = function() {
                  var first = arguments[0];
                  if (
                    typeof first === 'string' &&
                    (first.indexOf('bis_skin_checked') !== -1 || first.indexOf('chrome-extension://') !== -1)
                  ) {
                    return;
                  }
                  return _origError.apply(console, arguments);
                };

                // 3. Strip extension-injected DOM attributes (bis_skin_checked) before/during hydration
                try {
                  var obs = new MutationObserver(function(mutations) {
                    for (var i = 0; i < mutations.length; i++) {
                      var m = mutations[i];
                      if (m.type === 'attributes' && m.attributeName === 'bis_skin_checked' && m.target) {
                        m.target.removeAttribute('bis_skin_checked');
                      }
                    }
                  });
                  obs.observe(document.documentElement, {
                    attributes: true,
                    subtree: true,
                    attributeFilter: ['bis_skin_checked']
                  });
                } catch (e) {}
              })();
            `,
          }}
        />
      </head>
      <body className="font-sans antialiased" suppressHydrationWarning>
        {children}
        {process.env.NODE_ENV === 'production' && <Analytics />}
      </body>
    </html>
  )
}
